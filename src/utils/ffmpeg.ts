import { execSync, exec } from "child_process";
import { existsSync, mkdirSync, writeFileSync, unlinkSync } from "fs";
import { join, dirname } from "path";

/**
 * Check if ffmpeg is installed and available
 */
export function checkFfmpeg(): boolean {
  try {
    execSync("ffmpeg -version", { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

/**
 * Get the duration of an audio file in milliseconds
 */
export function getAudioDuration(filePath: string): number {
  if (!existsSync(filePath)) {
    throw new Error(`Audio file not found: ${filePath}`);
  }

  try {
    const result = execSync(
      `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${filePath}"`,
      { encoding: "utf-8" }
    );
    const seconds = parseFloat(result.trim());
    return Math.round(seconds * 1000);
  } catch (error) {
    throw new Error(
      `Failed to get audio duration: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Validate that an audio file exists and is readable
 */
export function validateAudioFile(filePath: string): {
  valid: boolean;
  duration_ms?: number;
  format?: string;
  error?: string;
} {
  if (!existsSync(filePath)) {
    return { valid: false, error: "File not found" };
  }

  try {
    const probeResult = execSync(
      `ffprobe -v error -show_entries format=duration,format_name -of json "${filePath}"`,
      { encoding: "utf-8" }
    );
    const probe = JSON.parse(probeResult);

    return {
      valid: true,
      duration_ms: Math.round(parseFloat(probe.format.duration) * 1000),
      format: probe.format.format_name,
    };
  } catch (error) {
    return {
      valid: false,
      error: `Invalid audio file: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

export interface ChapterMarker {
  title: string;
  start_ms: number;
}

/**
 * Concatenate multiple audio files into one
 */
export function concatenateAudioFiles(
  inputFiles: string[],
  outputPath: string,
  format: "mp3" | "wav" | "m4a" = "mp3"
): void {
  if (inputFiles.length === 0) {
    throw new Error("No input files provided");
  }

  // Validate all input files exist
  for (const file of inputFiles) {
    if (!existsSync(file)) {
      throw new Error(`Input file not found: ${file}`);
    }
  }

  // Ensure output directory exists
  const outputDir = dirname(outputPath);
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }

  // Create a file list for ffmpeg concat demuxer
  const listPath = join(outputDir, ".concat_list.txt");
  const listContent = inputFiles
    .map((f) => `file '${f.replace(/'/g, "'\\''")}'`)
    .join("\n");
  writeFileSync(listPath, listContent);

  try {
    // Set codec based on format
    let codec = "";
    switch (format) {
      case "mp3":
        codec = "-c:a libmp3lame -q:a 2";
        break;
      case "wav":
        codec = "-c:a pcm_s16le";
        break;
      case "m4a":
        codec = "-c:a aac -b:a 192k";
        break;
    }

    execSync(
      `ffmpeg -y -f concat -safe 0 -i "${listPath}" ${codec} "${outputPath}"`,
      { stdio: "pipe" }
    );
  } finally {
    // Clean up the list file
    if (existsSync(listPath)) {
      unlinkSync(listPath);
    }
  }
}

/**
 * Create an MP3 with chapter markers (ID3v2 chapters)
 */
export function createAudiobookWithChapters(
  inputFiles: string[],
  outputPath: string,
  chapters: ChapterMarker[],
  metadata?: {
    title?: string;
    artist?: string;
    album?: string;
  }
): void {
  if (inputFiles.length === 0) {
    throw new Error("No input files provided");
  }

  // First concatenate all files
  const outputDir = dirname(outputPath);
  const tempPath = join(outputDir, ".temp_concat.mp3");

  try {
    concatenateAudioFiles(inputFiles, tempPath, "mp3");

    // Create FFMETADATA file for chapters
    const metadataPath = join(outputDir, ".ffmetadata.txt");
    let metadataContent = ";FFMETADATA1\n";

    if (metadata?.title) {
      metadataContent += `title=${metadata.title}\n`;
    }
    if (metadata?.artist) {
      metadataContent += `artist=${metadata.artist}\n`;
    }
    if (metadata?.album) {
      metadataContent += `album=${metadata.album}\n`;
    }

    // Add chapter markers
    for (let i = 0; i < chapters.length; i++) {
      const chapter = chapters[i];
      const nextChapter = chapters[i + 1];
      const startMs = chapter.start_ms;
      // End is either the start of the next chapter or we need to probe the file for total duration
      const endMs = nextChapter
        ? nextChapter.start_ms
        : getAudioDuration(tempPath);

      metadataContent += `\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=${startMs}\nEND=${endMs}\ntitle=${chapter.title}\n`;
    }

    writeFileSync(metadataPath, metadataContent);

    try {
      // Apply metadata to the file
      execSync(
        `ffmpeg -y -i "${tempPath}" -i "${metadataPath}" -map_metadata 1 -c copy "${outputPath}"`,
        { stdio: "pipe" }
      );
    } finally {
      if (existsSync(metadataPath)) {
        unlinkSync(metadataPath);
      }
    }
  } finally {
    if (existsSync(tempPath)) {
      unlinkSync(tempPath);
    }
  }
}

/**
 * Convert audio file to a specific format
 */
export function convertAudioFormat(
  inputPath: string,
  outputPath: string,
  format: "mp3" | "wav" | "m4a" = "mp3"
): void {
  if (!existsSync(inputPath)) {
    throw new Error(`Input file not found: ${inputPath}`);
  }

  const outputDir = dirname(outputPath);
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }

  let codec = "";
  switch (format) {
    case "mp3":
      codec = "-c:a libmp3lame -q:a 2";
      break;
    case "wav":
      codec = "-c:a pcm_s16le";
      break;
    case "m4a":
      codec = "-c:a aac -b:a 192k";
      break;
  }

  execSync(`ffmpeg -y -i "${inputPath}" ${codec} "${outputPath}"`, {
    stdio: "pipe",
  });
}

/**
 * Normalize audio levels across files
 */
export function normalizeAudio(inputPath: string, outputPath: string): void {
  if (!existsSync(inputPath)) {
    throw new Error(`Input file not found: ${inputPath}`);
  }

  const outputDir = dirname(outputPath);
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }

  // Two-pass loudness normalization to -16 LUFS (podcast/audiobook standard)
  execSync(
    `ffmpeg -y -i "${inputPath}" -af loudnorm=I=-16:TP=-1.5:LRA=11 "${outputPath}"`,
    { stdio: "pipe" }
  );
}
