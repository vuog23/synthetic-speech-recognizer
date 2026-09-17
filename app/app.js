import { limits } from './config.js';
import { encodeWav, formatTime, toMono, waveformPeaks } from './audio.js';

const $ = id => document.getElementById(id);
const audio = new Audio();
audio.preload = 'auto';
audio.volume = 0.8;
const state = { tracks: [], active: null, busy: null, decoding: false, recording: false, requestingMic: false, held: false, mode: 'speech', nextTrack: 1 };
let audioContext, recorder, stream, analyser, liveSource, liveSamples;
let recordingTimer, recordingStarted, animationFrame, requestController;
let taskRevision = 0, peaks = null;

function setStatus(message, kind = '') {
  $('status-text').textContent = message;
  $('status-bar').className = `status-bar ${kind}`;
}

function updateControls() {
  const locked = Boolean(state.busy || state.decoding || state.recording || state.requestingMic);
  const hasAudio = Boolean(state.active);
  $('record-button').disabled = Boolean(state.busy || state.decoding);
  $('upload-button').disabled = locked;
  $('generate').disabled = locked || !$('transcript').value.trim() || $('transcript').value.length > limits.textCharacters;
  $('track-select').disabled = locked;
  $('classify').disabled = locked || !hasAudio;
  $('retry-transcribe').disabled = locked;
  $('transcript').readOnly = state.busy === 'transcribe';
  for (const id of ['play', 'replay', 'player-replay', 'seek', 'waveform-seek', 'download']) $(id).disabled = !hasAudio || state.recording || state.requestingMic || state.decoding;
  for (const id of ['speech-tab', 'text-tab']) $(id).disabled = state.recording || state.requestingMic;
  $('cancel-task').hidden = !state.busy;
  $('character-count').textContent = `${$('transcript').value.length.toLocaleString()} / 1,000`;
}

function setMode(mode, focus = false) {
  if (state.recording || state.requestingMic) return;
  state.mode = mode;
  for (const name of ['speech', 'text']) {
    const active = name === mode;
    $(`${name}-tab`).classList.toggle('active', active);
    $(`${name}-tab`).setAttribute('aria-selected', String(active));
    $(`${name}-tab`).tabIndex = active ? 0 : -1;
    $(`${name}-input`).hidden = !active;
  }
  if (mode === 'text' && focus) $('transcript').focus();
}
$('speech-tab').onclick = () => setMode('speech');
$('text-tab').onclick = () => setMode('text', true);
for (const name of ['speech', 'text']) $(`${name}-tab`).onkeydown = event => {
  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
    event.preventDefault();
    const mode = event.key === 'Home' ? 'speech' : event.key === 'End' ? 'text' : state.mode === 'speech' ? 'text' : 'speech';
    setMode(mode);
    $(`${mode}-tab`).focus();
  }
};

function getContext() {
  if (!audioContext) audioContext = new AudioContext();
  return audioContext;
}

function cancelRequest() {
  requestController?.abort();
  requestController = null;
}

async function infer(type, payload) {
  const controller = new AbortController();
  requestController = controller;
  const timeout = setTimeout(() => controller.abort('timeout'), 10 * 60 * 1000);
  try {
    const body = type === 'synthesize' ? JSON.stringify(payload) : encodeWav(payload.samples, payload.sampleRate);
    const response = await fetch('/api/' + type, {
      method: 'POST', signal: controller.signal, body,
      headers: { 'Content-Type': type === 'synthesize' ? 'application/json' : 'audio/wav' },
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      const detail = Array.isArray(error.detail) ? error.detail.map(item => item.msg).join(' ') : error.detail;
      throw new Error(detail || 'The server could not complete this request (HTTP ' + response.status + ').');
    }
    if (type === 'synthesize') {
      const decoder = new OfflineAudioContext(1, 1, 24000);
      const buffer = await decoder.decodeAudioData(await response.arrayBuffer());
      return { samples: toMono(buffer), sampleRate: buffer.sampleRate };
    }
    return await response.json();
  } catch (error) {
    if (controller.signal.reason === 'timeout') throw new Error('The server took too long. Check its terminal and retry.');
    if (error instanceof TypeError) throw new Error('Cannot reach the speech server. Start FastAPI with python run.py, then retry.');
    throw error;
  } finally {
    clearTimeout(timeout);
    if (requestController === controller) requestController = null;
  }
}

async function runTask(type, action) {
  if (state.busy) return;
  const revision = ++taskRevision;
  state.busy = type;
  updateControls();
  setStatus(type === 'transcribe' ? 'Preparing your recording for Whisper base…' : type === 'synthesize' ? 'Preparing Kokoro 82M…' : 'Preparing your audio for analysis…', 'busy');
  try {
    await action(() => revision === taskRevision);
  } catch (error) {
    if (revision === taskRevision) {
      console.error(error);
      setStatus(error.message || 'Something went wrong. Please try again.', 'error');
      if (type === 'transcribe') {
        $('retry-transcribe').hidden = false;
        $('transcript-state').textContent = 'TRANSCRIPTION FAILED';
      }
    }
  } finally {
    if (revision === taskRevision) { state.busy = null; updateControls(); }
  }
}

$('cancel-task').onclick = () => {
  const type = state.busy;
  ++taskRevision;
  cancelRequest();
  state.busy = null;
  if (type === 'transcribe') {
    $('retry-transcribe').hidden = false;
    $('transcript-state').textContent = 'TRANSCRIPTION CANCELED';
  }
  setStatus('Stopped waiting. Your audio and text are kept; the server may finish its current inference.');
  updateControls();
};

function renderTracks() {
  $('track-select').replaceChildren();
  $('track-select').hidden = state.tracks.length < 2;
  $('audio-name').hidden = state.tracks.length >= 2;
  for (const track of [...state.tracks].reverse()) {
    const option = document.createElement('option');
    option.value = state.tracks.indexOf(track);
    option.textContent = track.name;
    option.selected = track === state.active;
    $('track-select').append(option);
  }
}
$('track-select').onchange = event => selectTrack(state.tracks[Number(event.target.value)]);

function renderResult() {
  const result = state.active?.result;
  $('result-empty').hidden = Boolean(result);
  $('result-content').hidden = !result;
  if (!result) return;
  $('result-content').classList.toggle('synthetic', result.label === 'Synthetic');
  $('result-label').textContent = result.label;
  $('result-probability').textContent = `${(result.probability * 100).toFixed(1)}%`;
  $('result-scores').replaceChildren();
  result.labels.forEach((label, i) => {
    const row = document.createElement('div'); row.className = 'score-row';
    const heading = document.createElement('div'); heading.className = 'score-label';
    const name = document.createElement('span'); name.textContent = label;
    const score = document.createElement('span'); score.textContent = `${(result.scores[i] * 100).toFixed(1)}%`;
    heading.append(name, score);
    const bar = document.createElement('div'); bar.className = 'score-bar';
    const fill = document.createElement('span'); fill.style.width = `${result.scores[i] * 100}%`;
    bar.append(fill); row.append(heading, bar); $('result-scores').append(row);
  });
  $('classifier-name').textContent = result.model;
  $('result-track').textContent = `${state.active.name} · ${result.model}`;
}

function addTrack(samples, sampleRate, kind, name, text = '') {
  if (!samples.length || !samples.every(Number.isFinite)) throw new Error('The audio contains no valid samples.');
  const blob = encodeWav(samples, sampleRate);
  const track = { samples, sampleRate, kind, name, text, sourceText: text, duration: samples.length / sampleRate, url: URL.createObjectURL(blob), result: null, transcribed: kind === 'generated' };
  state.tracks.push(track);
  // Bound session memory while retaining the most recent eight takes.
  if (state.tracks.length > 8) URL.revokeObjectURL(state.tracks.shift().url);
  selectTrack(track);
  return track;
}

function selectTrack(track) {
  audio.pause();
  state.active = track;
  audio.src = track.url;
  $('transcript').value = track.text;
  $('editor-hint').textContent = track.kind === 'generated' && track.text !== track.sourceText ? 'Text changed. Generate again to update the audio.' : 'Click to edit. Make it sound like you.';
  $('transcript-state').textContent = track.kind === 'generated' ? 'SOURCE TEXT' : track.transcribed ? 'WHISPER BASE' : 'EDITABLE TEXT';
  $('retry-transcribe').hidden = track.kind === 'generated' || track.transcribed;
  $('audio-name').textContent = track.name;
  $('audio-duration').textContent = formatTime(track.duration);
  $('waveform-end').textContent = formatTime(track.duration);
  $('audio-format').textContent = `${(track.sampleRate / 1000).toFixed(1)} kHz · MONO`;
  $('player-title').textContent = track.name;
  $('player-subtitle').textContent = track.kind === 'generated' ? 'Kokoro 82M · Generated speech' : 'Your session · Original audio';
  $('total').textContent = formatTime(track.duration);
  $('waveform-empty').hidden = true;
  $('playhead').hidden = false;
  $('analysis-source').textContent = `Analyzing: ${track.name}`;
  peaks = null;
  renderTracks(); renderResult(); updateControls(); syncPlayback(); drawWaveform();
  if (!state.busy && !state.decoding) setStatus(`Selected ${track.name}. ${track.result ? `Prediction: ${track.result.label} · ${(track.result.probability * 100).toFixed(1)}%.` : 'Ready to play or analyze.'}`);
}

function drawWaveform() {
  const canvas = $('waveform');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr); peaks = null;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const count = Math.max(1, Math.floor(rect.width / 5));
  let bars;
  if (state.recording && analyser) {
    analyser.getFloatTimeDomainData(liveSamples);
    bars = waveformPeaks(liveSamples, count);
  } else if (state.active) {
    if (!peaks || peaks.length !== count) peaks = waveformPeaks(state.active.samples, count);
    bars = peaks;
  }
  let peak = 0.05;
  if (bars) for (const value of bars) peak = Math.max(peak, value);
  const progress = state.active ? audio.currentTime / state.active.duration : 0;
  for (let i = 0; i < count; i++) {
    const height = bars ? Math.max(2, bars[i] / peak * (rect.height - 12)) : 3;
    ctx.fillStyle = state.recording ? '#e9a794' : bars ? (i / count <= progress ? '#a8efb9' : '#4f9c68') : '#344d3c';
    ctx.fillRect(i * rect.width / count, (rect.height - height) / 2, 2.5, height);
  }
}

function syncPlayback() {
  const duration = state.active?.duration || 0;
  const progress = duration ? Math.min(1, audio.currentTime / duration) : 0;
  for (const id of ['seek', 'waveform-seek']) $(id).value = Math.round(progress * 1000);
  $('seek').style.setProperty('--progress', `${progress * 100}%`);
  $('playhead').style.left = `calc(${progress * 100}% - ${progress * 2}px)`;
  $('elapsed').textContent = formatTime(audio.currentTime);
  $('play-icon').setAttribute('href', audio.paused ? '#i-play' : '#i-pause');
  $('play').setAttribute('aria-label', audio.paused ? 'Play audio' : 'Pause audio');
}

function animate() {
  cancelAnimationFrame(animationFrame);
  syncPlayback(); drawWaveform();
  if (!audio.paused || state.recording) animationFrame = requestAnimationFrame(animate);
}
audio.addEventListener('play', animate);
audio.addEventListener('pause', animate);
audio.addEventListener('ended', animate);
audio.addEventListener('timeupdate', syncPlayback);
audio.addEventListener('error', () => { if (state.active) setStatus('This audio could not be played. Try recording or uploading it again.', 'error'); });
new ResizeObserver(drawWaveform).observe($('waveform-wrap'));

async function play(restart = false) {
  if (!state.active) return;
  try {
    if (restart) audio.currentTime = 0;
    await audio.play();
  } catch (error) { setStatus(`Playback could not start: ${error.message}`, 'error'); }
}
$('play').onclick = () => audio.paused ? play() : audio.pause();
$('replay').onclick = $('player-replay').onclick = () => play(true);
for (const id of ['seek', 'waveform-seek']) $(id).oninput = event => {
  if (state.active) { audio.currentTime = Number(event.target.value) / 1000 * state.active.duration; syncPlayback(); drawWaveform(); }
};
$('volume').style.setProperty('--progress', '80%');
$('volume').oninput = event => {
  audio.volume = Number(event.target.value);
  event.target.style.setProperty('--progress', `${audio.volume * 100}%`);
};
$('download').onclick = () => {
  if (!state.active) return;
  const link = document.createElement('a');
  link.href = state.active.url;
  link.download = `${state.active.name.replace(/[^a-z0-9_-]/gi, '_')}.wav`;
  link.click();
};

function releaseMicrophone() {
  clearInterval(recordingTimer);
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  liveSource?.disconnect(); liveSource = null; analyser = null;
  state.recording = false; state.requestingMic = false; state.held = false;
  $('record-zone').classList.remove('recording');
  $('record-title').textContent = 'Something to say?';
  $('record-hint').textContent = 'Hold the button, speak, then release. We’ll take care of the transcription.';
  $('record-time').textContent = 'Up to 60 seconds · Hold Space when focused';
  $('record-button').setAttribute('aria-label', 'Hold to record. Release to stop.');
  $('waveform-empty').hidden = Boolean(state.active);
  $('playhead').hidden = !state.active;
  updateControls(); animate();
}

async function startRecording() {
  if (state.busy || state.decoding || state.recording || state.requestingMic) return;
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setStatus('Recording needs a browser with microphone support on localhost or HTTPS. You can also upload audio.', 'error');
    return;
  }
  state.held = true; state.requestingMic = true; audio.pause(); updateControls();
  setStatus('Allow microphone access, then keep holding to record.');
  try {
    const context = getContext();
    await context.resume();
    const acquired = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    if (!state.held) {
      acquired.getTracks().forEach(track => track.stop());
      releaseMicrophone();
      setStatus('Microphone ready. Hold the button again to record.');
      return;
    }
    stream = acquired;
    const mimeType = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus'].find(type => MediaRecorder.isTypeSupported(type));
    recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    const chunks = [];
    let failed = false;
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onerror = event => { failed = true; releaseMicrophone(); setStatus(event.error?.message || 'Recording failed. Please try again.', 'error'); };
    recorder.onstop = async () => {
      const duration = (performance.now() - recordingStarted) / 1000;
      const type = recorder.mimeType;
      releaseMicrophone();
      if (failed) return;
      if (duration < 0.2 || !chunks.length) { setStatus('That was a little short. Hold the button for at least a moment and speak.', 'error'); return; }
      await loadAudio(new Blob(chunks, { type }), `Recording ${String(state.nextTrack++).padStart(2, '0')}`);
    };
    analyser = context.createAnalyser(); analyser.fftSize = 2048;
    liveSamples = new Float32Array(analyser.fftSize);
    liveSource = context.createMediaStreamSource(stream); liveSource.connect(analyser);
    recorder.start(150);
    recordingStarted = performance.now();
    state.recording = true; state.requestingMic = false;
    $('record-zone').classList.add('recording');
    $('record-title').textContent = 'We’re listening.';
    $('record-hint').textContent = 'Release the button when you’re done. Your words will appear below.';
    $('record-button').setAttribute('aria-label', 'Recording. Release to stop.');
    $('waveform-empty').hidden = true; $('playhead').hidden = true;
    setStatus('Recording… release to finish.'); updateControls(); animate();
    recordingTimer = setInterval(() => {
      const seconds = (performance.now() - recordingStarted) / 1000;
      $('record-time').textContent = `● ${formatTime(seconds)} / 1:00 · Recording`;
      if (seconds >= limits.recordingSeconds) stopRecording();
    }, 100);
  } catch (error) {
    releaseMicrophone();
    const messages = { NotAllowedError: 'Microphone access was denied. Enable it in your browser’s site settings, then try again.', NotFoundError: 'No microphone was found. Connect one or upload an audio file.', NotReadableError: 'The microphone is in use or unavailable. Check your device and try again.' };
    setStatus(messages[error.name] || `Recording could not start: ${error.message}`, 'error');
  }
}

function stopRecording() {
  state.held = false;
  if (recorder?.state === 'recording') recorder.stop();
}
$('record-button').onpointerdown = event => {
  if (event.button !== 0) return;
  event.preventDefault();
  $('record-button').focus();
  $('record-button').setPointerCapture(event.pointerId);
  startRecording();
};
window.addEventListener('pointerup', stopRecording);
window.addEventListener('pointercancel', stopRecording);
$('record-button').addEventListener('lostpointercapture', stopRecording);
$('record-button').oncontextmenu = event => event.preventDefault();
$('record-button').onkeydown = event => {
  if ([' ', 'Enter'].includes(event.key)) { event.preventDefault(); if (!event.repeat) startRecording(); }
};
window.addEventListener('keyup', event => { if ([' ', 'Enter'].includes(event.key) && state.held) stopRecording(); });
window.addEventListener('blur', stopRecording);
document.addEventListener('visibilitychange', () => { if (document.hidden) stopRecording(); });

async function loadAudio(blob, name) {
  state.decoding = true; updateControls(); setStatus('Preparing your waveform…', 'busy');
  let track;
  try {
    if (blob.size > limits.uploadBytes) throw new Error('Choose an audio file smaller than 25 MB.');
    const decoded = await getContext().decodeAudioData(await blob.arrayBuffer());
    if (decoded.duration > limits.recordingSeconds + 0.5) throw new Error('Please use audio of 60 seconds or less.');
    if (decoded.duration < 0.2) throw new Error('Please use audio at least 0.2 seconds long.');
    track = addTrack(toMono(decoded), decoded.sampleRate, 'recorded', name);
  } catch (error) {
    setStatus(error.name === 'EncodingError' ? 'This file could not be decoded. Try a WAV, MP3, or WebM audio file.' : error.message, 'error');
  } finally { state.decoding = false; updateControls(); }
  if (track) await transcribe(track);
}
$('upload-button').onclick = () => $('audio-upload').click();
$('audio-upload').onchange = async event => {
  const file = event.target.files[0]; event.target.value = '';
  if (!file || state.busy || state.recording || state.requestingMic || state.decoding) return;
  audio.pause();
  await loadAudio(file, file.name.replace(/\.[^.]+$/, ''));
};

async function transcribe(track) {
  await runTask('transcribe', async isCurrent => {
    $('transcript-state').textContent = 'TRANSCRIBING…';
    const result = await infer('transcribe', { samples: track.samples, sampleRate: track.sampleRate });
    if (!isCurrent()) return;
    track.text = result.text; track.transcribed = true;
    $('transcript').value = result.text;
    $('transcript-state').textContent = 'WHISPER BASE';
    $('retry-transcribe').hidden = true;
    setStatus(result.text ? 'Transcription ready. Edit the words, generate a new voice, or analyze your recording.' : 'No words were detected. You can record again, type text, or analyze this audio.');
  });
}
$('retry-transcribe').onclick = () => { if (state.active) transcribe(state.active); };
$('transcript').oninput = () => {
  if (state.active) state.active.text = $('transcript').value;
  $('editor-hint').textContent = $('transcript').value.length > limits.textCharacters ? 'Shorten to 1,000 characters to generate speech.' : state.active?.kind === 'generated' && $('transcript').value !== state.active.sourceText ? 'Text changed. Generate again to update the audio.' : 'Click to edit. Make it sound like you.';
  updateControls();
};

$('generate').onclick = () => {
  const text = $('transcript').value.trim();
  const voice = $('voice').value, speed = Number($('speed').value);
  if (!text || text.length > limits.textCharacters) return;
  runTask('synthesize', async isCurrent => {
    const result = await infer('synthesize', { text, voice, speed });
    if (!isCurrent()) return;
    addTrack(result.samples, result.sampleRate, 'generated', `Generated ${String(state.nextTrack++).padStart(2, '0')}`, text);
    setStatus('Your new voice is ready. Press play to listen, or Analyze speech to classify it.');
  });
};
$('classify').onclick = () => {
  const track = state.active;
  if (!track) return;
  runTask('classify', async isCurrent => {
    track.result = null; renderResult();
    const result = await infer('classify', { samples: track.samples, sampleRate: track.sampleRate });
    if (!isCurrent()) return;
    track.result = result; renderResult();
    const seconds = result.windowSeconds;
    const crop = track.duration > seconds ? ` The model analyzes the center ${seconds} seconds of this clip.` : '';
    setStatus(`Analysis complete: ${result.label} · ${(result.probability * 100).toFixed(1)}%.${crop}`);
  });
};

window.addEventListener('pagehide', () => { stopRecording(); releaseMicrophone(); cancelRequest(); audioContext?.close(); state.tracks.forEach(track => URL.revokeObjectURL(track.url)); });
updateControls(); drawWaveform();
fetch('/api/health').then(response => {
  if (!response.ok) throw new Error('Server unavailable');
  return response.json();
}).then(health => { $('classifier-name').textContent = health.classifier; })
  .catch(() => setStatus('Cannot reach FastAPI. Start the server with python run.py, then reload.', 'error'));
