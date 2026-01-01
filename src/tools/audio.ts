import { existsSync, copyFileSync, mkdirSync } from "fs";
import { join, basename, extname } from "path";
import { getDatabase, getAudiobookDir, getCurrentProjectPath } from "../db/connection.js";
import type { Segment } from "../db/schema.js";
import { getChapter, listChapters, getChaptersWithStats } from "./chapters.js";
import { listSegments, getSegment, getPendingSegments } from "./segments.js";
import { getProjectInfo } from "./projects.js";
import {
  checkFfmpeg,
  validateAudioFile,
  getAudioDuration,
  concatenateAudioFiles,
  createAudiobookWithChapters,
  type ChapterMarker,
} from "../utils/ffmpeg.js";

export interface RegisterSegmentAudioParams {
  segment_id: string;
  audio_path: string;
  duration_ms?: number;
}

export interface StitchChapterParams {
  chapter_id: string;
  output_filename?: string;
}

export interface StitchBookParams {
  output_filename?: string;
  include_chapter_markers?: boolean;
}

/**
 * Register an audio file for a segment
 */
export function registerSegmentAudio(params: RegisterSegmentAudioParams): {
  segment_id: string;
  audio_path: string;
  duration_ms: number;
  copied_to: string;
} {
  const { segment_id, audio_path, duration_ms } = params;
  const db = getDatabase();
  const projectPath = getCurrentProjectPath();

  if (!projectPath) {
    throw new Error("No project is currently open");
  }

  // Verify segment exists
  const segment = getSegment(segment_id);
  if (!segment) {
    throw new Error(`Segment not found: ${segment_id}`);
  }

  // Validate audio file
  const validation = validateAudioFile(audio_path);
  if (!validation.valid) {
    throw new Error(`Invalid audio file: ${validation.error}`);
  }

  // Use provided duration or detected duration
  const actualDuration = duration_ms ?? validation.duration_ms!;

  // Copy audio to project's audio directory
  const audiobookDir = getAudiobookDir(projectPath);
  const segmentsDir = join(audiobookDir, "audio", "segments");
  if (!existsSync(segmentsDir)) {
    mkdirSync(segmentsDir, { recursive: true });
  }

  const ext = extname(audio_path) || ".mp3";
  const destPath = join(segmentsDir, `${segment_id}${ext}`);
  copyFileSync(audio_path, destPath);

  // Store relative path in database
  const relativePath = `audio/segments/${segment_id}${ext}`;

  // Update segment with audio info
  const stmt = db.prepare(`
    UPDATE segments SET audio_path = ?, duration_ms = ? WHERE id = ?
  `);
  stmt.run(relativePath, actualDuration, segment_id);

  return {
    segment_id,
    audio_path: relativePath,
    duration_ms: actualDuration,
    copied_to: destPath,
  };
}

/**
 * Get the status of audio for a chapter
 */
export function getChapterAudioStatus(chapter_id: string): {
  chapter_id: string;
  chapter_title: string;
  total_segments: number;
  segments_with_audio: number;
  segments_missing_audio: number;
  total_duration_ms: number;
  ready_to_stitch: boolean;
  missing_segments: Array<{
    id: string;
    sort_order: number;
    text_preview: string;
    character_name: string | null;
  }>;
} {
  const db = getDatabase();

  // Verify chapter exists
  const chapter = getChapter(chapter_id);
  if (!chapter) {
    throw new Error(`Chapter not found: ${chapter_id}`);
  }

  const segments = listSegments(chapter_id);
  const withAudio = segments.filter((s) => s.audio_path);
  const missingAudio = segments.filter((s) => !s.audio_path);

  // Get character names for missing segments
  const missingWithNames = missingAudio.map((s) => {
    let characterName: string | null = null;
    if (s.character_id) {
      const charResult = db
        .prepare(`SELECT name FROM characters WHERE id = ?`)
        .get(s.character_id) as { name: string } | undefined;
      characterName = charResult?.name ?? null;
    }
    return {
      id: s.id,
      sort_order: s.sort_order,
      text_preview:
        s.text_content.length > 50
          ? s.text_content.slice(0, 50) + "..."
          : s.text_content,
      character_name: characterName,
    };
  });

  const totalDuration = withAudio.reduce(
    (sum, s) => sum + (s.duration_ms ?? 0),
    0
  );

  return {
    chapter_id,
    chapter_title: chapter.title,
    total_segments: segments.length,
    segments_with_audio: withAudio.length,
    segments_missing_audio: missingAudio.length,
    total_duration_ms: totalDuration,
    ready_to_stitch: missingAudio.length === 0 && segments.length > 0,
    missing_segments: missingWithNames,
  };
}

/**
 * Stitch all segments in a chapter into a single audio file
 */
export function stitchChapter(params: StitchChapterParams): {
  chapter_id: string;
  chapter_title: string;
  output_path: string;
  segment_count: number;
  total_duration_ms: number;
} {
  const { chapter_id, output_filename } = params;
  const projectPath = getCurrentProjectPath();

  if (!projectPath) {
    throw new Error("No project is currently open");
  }

  // Check ffmpeg availability
  if (!checkFfmpeg()) {
    throw new Error("ffmpeg is not installed or not in PATH");
  }

  // Verify chapter exists and has all audio
  const status = getChapterAudioStatus(chapter_id);
  if (!status.ready_to_stitch) {
    throw new Error(
      `Chapter is not ready to stitch. Missing ${status.segments_missing_audio} audio files.`
    );
  }

  const audiobookDir = getAudiobookDir(projectPath);
  const segments = listSegments(chapter_id);

  // Get absolute paths for all audio files
  const inputFiles = segments.map((s) => join(audiobookDir, s.audio_path!));

  // Determine output path
  const filename =
    output_filename ||
    `${status.chapter_title.replace(/[^a-zA-Z0-9]/g, "_")}.mp3`;
  const outputDir = join(audiobookDir, "exports", "chapters");
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }
  const outputPath = join(outputDir, filename);

  // Concatenate files
  concatenateAudioFiles(inputFiles, outputPath, "mp3");

  // Get actual duration of output
  const duration = getAudioDuration(outputPath);

  return {
    chapter_id,
    chapter_title: status.chapter_title,
    output_path: outputPath,
    segment_count: segments.length,
    total_duration_ms: duration,
  };
}

/**
 * Get the overall stitch status for the book
 */
export function getStitchStatus(): {
  total_chapters: number;
  chapters_ready: number;
  total_segments: number;
  segments_with_audio: number;
  total_duration_ms: number;
  ready_to_stitch_book: boolean;
  chapters: Array<{
    id: string;
    title: string;
    sort_order: number;
    segment_count: number;
    segments_with_audio: number;
    duration_ms: number;
    ready: boolean;
  }>;
} {
  const chapters = getChaptersWithStats();

  const chapterStatuses = chapters.map((ch) => {
    const status = getChapterAudioStatus(ch.id);
    return {
      id: ch.id,
      title: ch.title,
      sort_order: ch.sort_order,
      segment_count: ch.segment_count,
      segments_with_audio: status.segments_with_audio,
      duration_ms: status.total_duration_ms,
      ready: status.ready_to_stitch,
    };
  });

  const totalSegments = chapterStatuses.reduce(
    (sum, ch) => sum + ch.segment_count,
    0
  );
  const totalWithAudio = chapterStatuses.reduce(
    (sum, ch) => sum + ch.segments_with_audio,
    0
  );
  const totalDuration = chapterStatuses.reduce(
    (sum, ch) => sum + ch.duration_ms,
    0
  );
  const chaptersReady = chapterStatuses.filter((ch) => ch.ready).length;

  return {
    total_chapters: chapters.length,
    chapters_ready: chaptersReady,
    total_segments: totalSegments,
    segments_with_audio: totalWithAudio,
    total_duration_ms: totalDuration,
    ready_to_stitch_book:
      chaptersReady === chapters.length && chapters.length > 0,
    chapters: chapterStatuses,
  };
}

/**
 * Stitch all chapters into a complete audiobook
 */
export function stitchBook(params: StitchBookParams): {
  output_path: string;
  chapter_count: number;
  total_duration_ms: number;
  chapters: Array<{
    title: string;
    start_ms: number;
  }>;
} {
  const { output_filename, include_chapter_markers = true } = params;
  const projectPath = getCurrentProjectPath();

  if (!projectPath) {
    throw new Error("No project is currently open");
  }

  // Check ffmpeg availability
  if (!checkFfmpeg()) {
    throw new Error("ffmpeg is not installed or not in PATH");
  }

  // Check overall status
  const status = getStitchStatus();
  if (!status.ready_to_stitch_book) {
    const notReady = status.chapters.filter((ch) => !ch.ready);
    throw new Error(
      `Book is not ready to stitch. Chapters not ready: ${notReady.map((ch) => ch.title).join(", ")}`
    );
  }

  const audiobookDir = getAudiobookDir(projectPath);
  const info = getProjectInfo();
  const chapters = listChapters();

  // First, stitch each chapter and collect the chapter files
  const chapterFiles: string[] = [];
  const chapterMarkers: ChapterMarker[] = [];
  let currentMs = 0;

  for (const chapter of chapters) {
    // Stitch the chapter
    const result = stitchChapter({ chapter_id: chapter.id });
    chapterFiles.push(result.output_path);

    // Record chapter marker
    chapterMarkers.push({
      title: chapter.title,
      start_ms: currentMs,
    });

    currentMs += result.total_duration_ms;
  }

  // Determine output path
  const filename =
    output_filename ||
    `${info.project.title.replace(/[^a-zA-Z0-9]/g, "_")}.mp3`;
  const outputDir = join(audiobookDir, "exports", "book");
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }
  const outputPath = join(outputDir, filename);

  // Create final audiobook
  if (include_chapter_markers) {
    createAudiobookWithChapters(chapterFiles, outputPath, chapterMarkers, {
      title: info.project.title,
      artist: info.project.author ?? undefined,
      album: info.project.title,
    });
  } else {
    concatenateAudioFiles(chapterFiles, outputPath, "mp3");
  }

  // Get actual duration
  const duration = getAudioDuration(outputPath);

  return {
    output_path: outputPath,
    chapter_count: chapters.length,
    total_duration_ms: duration,
    chapters: chapterMarkers,
  };
}

/**
 * Clear audio from a segment
 */
export function clearSegmentAudio(segment_id: string): void {
  const db = getDatabase();

  // Verify segment exists
  const segment = getSegment(segment_id);
  if (!segment) {
    throw new Error(`Segment not found: ${segment_id}`);
  }

  // Clear audio path and duration
  const stmt = db.prepare(`
    UPDATE segments SET audio_path = NULL, duration_ms = NULL WHERE id = ?
  `);
  stmt.run(segment_id);
}
