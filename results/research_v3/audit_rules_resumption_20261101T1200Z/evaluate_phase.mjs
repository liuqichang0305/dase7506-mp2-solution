// Execute only the immutable official rule module with whitelisted metadata.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {submissionPhase, reviewEndTime, PROJECTS} from './arena.mjs';
const base = path.dirname(fileURLToPath(import.meta.url));
const meta = JSON.parse(fs.readFileSync(path.join(base, 'publication_metadata_only.json'), 'utf8'));
const now = new Date();
const result = {
  checked_at_utc: now.toISOString(),
  official_site_commit: '65c16a3eef424e30b254ff58804dffe0fbda8793',
  project: 'mp2',
  phase: submissionPhase('mp2', {projects: meta.projects}, +now),
  deadline: PROJECTS.mp2.score_deadline,
  deadline_utc: new Date(PROJECTS.mp2.score_deadline).toISOString(),
  published_at: meta.projects.mp2.published_at,
  review_ends_at: meta.projects.mp2.review_ends_at,
  review_ends_at_utc: new Date(reviewEndTime('mp2', meta.projects.mp2)).toISOString(),
  sources: ['arena.mjs', 'projects.mjs', 'publication_metadata_only.json'],
  submission_entries_loaded: false,
  network_operations: false
};
fs.writeFileSync(path.join(base, 'phase_result.json'), JSON.stringify(result, null, 2)+'\n', {flag:'wx'});
console.log(JSON.stringify(result, null, 2));
