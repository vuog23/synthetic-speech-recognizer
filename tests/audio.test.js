import test from 'node:test';
import assert from 'node:assert/strict';
import { encodeWav, toMono, waveformPeaks } from '../app/audio.js';

test('mono downmix averages channels, waveform peaks retain short transients', () => {
  const mono = toMono({ length: 3, numberOfChannels: 2, getChannelData: i => [Float32Array.of(1, 0, -1), Float32Array.of(0, 1, 1)][i] });
  assert.deepEqual([...mono], [0.5, 0.5, 0]);
  assert.deepEqual([...waveformPeaks(Float32Array.of(0, 1, 0, -0.8), 2)], [1, Math.fround(0.8)]);
});

test('WAV export encodes a mono PCM16 header and clips safely', async () => {
  const blob = encodeWav(Float32Array.of(-2, 0, 2), 24000);
  const view = new DataView(await blob.arrayBuffer());
  assert.equal(blob.type, 'audio/wav');
  assert.equal(view.byteLength, 50);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 24000);
  assert.equal(view.getInt16(44, true), -32768);
  assert.equal(view.getInt16(46, true), 0);
  assert.equal(view.getInt16(48, true), 32767);
});
