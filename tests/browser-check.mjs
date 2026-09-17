// Install Playwright into .cache/browser-tests and set CHROME_PATH before running.
import { chromium } from '../.cache/browser-tests/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { encodeWav } from '../app/audio.js';

await mkdir('.cache/screenshots', { recursive: true });
const url = process.env.BASE_URL || 'http://localhost:3000';
const context = await chromium.launchPersistentContext('.cache/browser-profile', {
  executablePath: process.env.CHROME_PATH, headless: true,
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  viewport: { width: 1440, height: 1100 }, permissions: ['microphone'],
});
try {
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const generated = Buffer.from(await encodeWav(Float32Array.from({ length: 48000 }, (_, i) => Math.sin(i / 10) * 0.4), 24000).arrayBuffer());
  let delay = 150;
  // Only inference responses are stubbed; static files and health come from FastAPI.
  await page.route('**/api/transcribe', async route => {
    assert.equal(route.request().headers()['content-type'], 'audio/wav');
    assert.equal(route.request().postDataBuffer().toString('ascii', 0, 4), 'RIFF');
    await new Promise(resolve => setTimeout(resolve, delay));
    await route.fulfill({ json: { text: 'This is a browser interaction test.' } });
  });
  await page.route('**/api/synthesize', async route => {
    assert.ok(route.request().postDataJSON().text);
    await new Promise(resolve => setTimeout(resolve, delay));
    await route.fulfill({ contentType: 'audio/wav', body: generated });
  });
  await page.route('**/api/classify', async route => {
    await new Promise(resolve => setTimeout(resolve, delay));
    await route.fulfill({ json: { label: 'Synthetic', probability: 0.75, scores: [0.25, 0.75], labels: ['Human', 'Synthetic'], model: 'Test classifier', windowSeconds: 3 } });
  });
  await page.goto(url);
  for (const selector of ['.sidebar', '.results-column', '.workflow-card', '.breadcrumb', '#new-session', '#settings-dialog', '#open-settings']) assert.equal(await page.locator(selector).count(), 0);
  await page.screenshot({ path: '.cache/screenshots/fastapi-desktop.png', fullPage: true });
  assert.equal(await page.locator('#generate').isDisabled(), true);
  assert.equal(await page.locator('#classify').isDisabled(), true);
  await page.locator('#text-tab').click();
  await page.locator('#transcript').fill('Every voice has a story. Let us listen closer.');
  assert.equal(await page.locator('#generate').isEnabled(), true);
  await page.locator('#speech-tab').click();
  const button = await page.locator('#record-button').boundingBox();
  await page.mouse.move(button.x + button.width / 2, button.y + button.height / 2);
  await page.mouse.down();
  await page.waitForFunction(() => document.querySelector('#record-zone').classList.contains('recording'));
  await page.waitForTimeout(1100);
  await page.mouse.up();
  await page.waitForFunction(() => document.querySelector('#transcript').value === 'This is a browser interaction test.');
  await page.locator('#replay').click();
  await page.waitForTimeout(400);
  assert.notEqual(await page.locator('#playhead').evaluate(el => el.style.left), 'calc(0% - 0px)');
  await page.locator('#player-replay').click();
  await page.locator('#generate').click();
  await page.waitForFunction(() => document.querySelector('#track-select').options.length === 2);
  await page.locator('#classify').click();
  await page.waitForFunction(() => document.querySelector('#result-probability').textContent === '75.0%');
  await page.screenshot({ path: '.cache/screenshots/fastapi-results.png', fullPage: true });
  await page.locator('#track-select').selectOption('0');
  assert.equal(await page.locator('#result-empty').isVisible(), true);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#track-select').selectOption('1');
  assert.equal(await page.locator('#result-label').textContent(), 'Synthetic');
  await page.screenshot({ path: '.cache/screenshots/fastapi-mobile.png', fullPage: true });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.reload();
  assert.equal(await page.locator('#classify').isDisabled(), true);
  await page.evaluate(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    window.testStreams = [];
    navigator.mediaDevices.getUserMedia = async constraints => {
      await new Promise(resolve => setTimeout(resolve, 400));
      const stream = await original(constraints);
      window.testStreams.push(stream);
      return stream;
    };
  });
  await page.locator('#record-button').scrollIntoViewIfNeeded();
  const pendingButton = await page.locator('#record-button').boundingBox();
  await page.mouse.move(pendingButton.x + 30, pendingButton.y + 30);
  await page.mouse.down();
  await page.waitForTimeout(80);
  await page.mouse.up();
  await page.waitForFunction(() => document.querySelector('#status-text').textContent.includes('Hold the button again'));
  assert.equal(await page.evaluate(() => window.testStreams.every(stream => stream.getTracks().every(track => track.readyState === 'ended'))), true);
  await page.evaluate(() => { navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Denied', 'NotAllowedError'); }; });
  await page.locator('#record-button').focus();
  await page.keyboard.down('Space');
  await page.waitForFunction(() => document.querySelector('#status-text').textContent.includes('access was denied'));
  await page.keyboard.up('Space');
  assert.equal(await page.locator('#record-button').isEnabled(), true);
  delay = 1200;
  await page.locator('#transcript').fill('Cancel this test.');
  await page.locator('#generate').click();
  await page.locator('#cancel-task').click();
  await page.waitForTimeout(1400);
  assert.equal(await page.locator('#track-select option').count(), 0);
  assert.equal(await page.locator('#generate').isEnabled(), true);
  assert.deepEqual(errors, []);
  console.log('PASS: simplified UI, recording/WAV requests, automatic STT, TTS/WAV response, classification, track isolation, replay, mobile, microphone permissions, cancellation. Inference responses stubbed.');

  if (process.env.REAL_MODELS === '1') {
    const real = await context.newPage();
    await real.goto(url);
    await real.locator('#text-tab').click();
    await real.locator('#transcript').fill('Every voice has a story. Let us listen closer.');
    console.log('Running actual FastAPI Kokoro synthesis…');
    await real.locator('#generate').click();
    await real.waitForFunction(() => document.querySelector('#player-title').textContent.startsWith('Generated') || document.querySelector('#status-bar').classList.contains('error'), null, { timeout: 600000 });
    assert.equal(await real.locator('#status-bar').evaluate(el => el.classList.contains('error')), false, await real.locator('#status-text').textContent());
    console.log('Running actual FastAPI ONNX classification…');
    await real.locator('#classify').click();
    await real.waitForFunction(() => !document.querySelector('#result-content').hidden || document.querySelector('#status-bar').classList.contains('error'), null, { timeout: 300000 });
    assert.equal(await real.locator('#result-content').isVisible(), true, await real.locator('#status-text').textContent());
    console.log('Actual classifier:', await real.locator('#result-label').textContent(), await real.locator('#result-probability').textContent());
    const downloadEvent = real.waitForEvent('download');
    await real.locator('#download').click();
    await (await downloadEvent).saveAs('.cache/generated-api-test.wav');
    await real.locator('#speech-tab').click();
    console.log('Uploading generated audio to actual FastAPI Whisper…');
    await real.locator('#audio-upload').setInputFiles('.cache/generated-api-test.wav');
    await real.waitForFunction(() => document.querySelector('#transcript-state').textContent === 'WHISPER BASE' || document.querySelector('#status-bar').classList.contains('error'), null, { timeout: 600000 });
    assert.match((await real.locator('#transcript').inputValue()).toLowerCase(), /voice/, await real.locator('#status-text').textContent());
    console.log('Actual Whisper:', await real.locator('#transcript').inputValue());
    await real.screenshot({ path: '.cache/screenshots/fastapi-real-models.png', fullPage: true });
  }
} finally { await context.close(); }
