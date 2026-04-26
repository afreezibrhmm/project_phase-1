import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS
# ============================================================
FS           = 100
BREATH_RATE  = 0.3    # Hz
HEART_RATE   = 1.2    # Hz
BUFFER_SIZE  = 300    # samples shown in live plots
NOISE_LEVEL  = 0.15

# ============================================================
# SIGNAL GENERATORS
# ============================================================
def get_human_sample(t, noise=NOISE_LEVEL):
    breath   = 0.8  * np.sin(2 * np.pi * BREATH_RATE * t)
    heart    = 0.15 * np.sin(2 * np.pi * HEART_RATE  * t)
    micro    = 0.05 * np.sin(2 * np.pi * 0.7 * t)
    multipath= 0.15 * np.sin(2 * np.pi * BREATH_RATE * t + 0.5)
    noise_v  = noise * np.random.randn()
    return breath + heart + micro + multipath + noise_v

def get_empty_sample(t, noise=0.08):
    static = 0.05 * np.sin(2 * np.pi * 0.05 * t)
    return static + noise * np.random.randn()

def bandpass_filter(signal, lowcut=0.1, highcut=2.5, fs=FS):
    if len(signal) < 20:
        return signal
    nyq = fs / 2
    b, a = butter(4, [lowcut/nyq, min(highcut/nyq, 0.99)], btype='band')
    return filtfilt(b, a, signal)

def fake_cnn_confidence(signal_buffer, is_human_phase):
    """Simulate CNN confidence based on signal variance"""
    if len(signal_buffer) < 10:
        return 50.0
    var = np.var(signal_buffer)
    if is_human_phase:
        conf = min(99.9, 75 + var * 120 + np.random.uniform(-3, 3))
    else:
        conf = max(0.1, 15 - var * 80 + np.random.uniform(-3, 3))
    return float(np.clip(conf, 0, 100))

# ============================================================
# FIGURE SETUP
# ============================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#080808')
fig.suptitle('🔴  LIVE SIMULATION — AI Human Detection Under Debris | UWB Radar System',
             fontsize=14, fontweight='bold', color='white', y=0.99)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                        left=0.05, right=0.97, top=0.93, bottom=0.06)

# Axes
ax_scene    = fig.add_subplot(gs[:, 0])      # scene diagram (left, full height)
ax_raw      = fig.add_subplot(gs[0, 1:3])    # raw signal
ax_fft      = fig.add_subplot(gs[1, 1:3])    # FFT
ax_breath   = fig.add_subplot(gs[2, 1:3])    # breathing extracted
ax_status   = fig.add_subplot(gs[0, 3])      # detection status
ax_conf     = fig.add_subplot(gs[1, 3])      # confidence history
ax_vitals   = fig.add_subplot(gs[2, 3])      # vital signs indicator

PURPLE='#7F77DD'; TEAL='#1D9E75'; AMBER='#EF9F27'
CORAL='#D85A30';  BLUE='#4FC3F7'; RED='#E24B4A'
GREEN='#00FF88'

def style(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#0f0f0f')
    ax.set_title(title, color='white', fontsize=9, pad=5, fontweight='bold')
    ax.tick_params(colors='#666666', labelsize=7)
    ax.grid(alpha=0.12, color='#333333')
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')
    if xlabel: ax.set_xlabel(xlabel, color='#666666', fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color='#666666', fontsize=7)

style(ax_raw,    '📡 Live UWB Radar Signal', 'Sample', 'Amplitude')
style(ax_fft,    '📊 FFT Frequency Spectrum', 'Frequency (Hz)', 'Magnitude')
style(ax_breath, '🫁 Extracted Vital Signs', 'Sample', 'Amplitude')
style(ax_conf,   '📈 Detection Confidence History', 'Frame', 'Confidence %')
style(ax_vitals, '❤️  Vital Signs Monitor')
ax_scene.set_facecolor('#0f0f0f')
ax_scene.set_xlim(0, 10); ax_scene.set_ylim(0, 14)
ax_scene.axis('off')
ax_status.set_facecolor('#0f0f0f')
ax_status.axis('off')

# ── STATIC SCENE ELEMENTS ──
# Title
ax_scene.text(5, 13.5, 'RADAR SCENE', ha='center', fontsize=9,
              color='#888888', fontweight='bold')

# Ground line
ax_scene.axhline(2.5, color='#555555', linewidth=2, xmin=0.05, xmax=0.95)
ax_scene.text(5, 2.2, 'Ground Level', ha='center', fontsize=7, color='#555555')

# Radar unit (left)
radar_box = FancyBboxPatch((0.3, 8.5), 2.2, 2.5,
                            boxstyle="round,pad=0.1",
                            facecolor='#1a2a3a', edgecolor=BLUE, linewidth=2)
ax_scene.add_patch(radar_box)
ax_scene.text(1.4, 10.5, '📡', ha='center', fontsize=18)
ax_scene.text(1.4, 9.5,  'UWB RADAR', ha='center', fontsize=7,
              color=BLUE, fontweight='bold')
ax_scene.text(1.4, 9.0,  'TX + RX', ha='center', fontsize=7, color='#888888')

# Debris block
debris_box = FancyBboxPatch((3.2, 2.5), 3.5, 7.5,
                             boxstyle="round,pad=0.05",
                             facecolor='#2a1800', edgecolor='#8B6914', linewidth=2)
ax_scene.add_patch(debris_box)
ax_scene.text(4.95, 9.6,  '🧱', ha='center', fontsize=20)
ax_scene.text(4.95, 8.8,  'DEBRIS', ha='center', fontsize=8,
              color='#8B6914', fontweight='bold')
ax_scene.text(4.95, 8.3,  '~1.5m thick', ha='center', fontsize=7, color='#666666')
# Debris texture dots
for dx in [3.6, 4.2, 4.8, 5.4, 6.0, 6.4]:
    for dy in [3.5, 5.0, 6.5, 7.5]:
        ax_scene.plot(dx, dy, 'o', color='#5a3a00', markersize=6, alpha=0.5)

# Human (right side of debris)
human_box = FancyBboxPatch((7.0, 2.5), 2.5, 5.0,
                            boxstyle="round,pad=0.1",
                            facecolor='#0a1a0a', edgecolor='#1D9E75', linewidth=2)
ax_scene.add_patch(human_box)
ax_scene.text(8.25, 6.8, '🧍', ha='center', fontsize=28)
ax_scene.text(8.25, 4.5, 'SURVIVOR', ha='center', fontsize=7,
              color=TEAL, fontweight='bold')
ax_scene.text(8.25, 4.0, 'Breathing\n+ Heartbeat', ha='center', fontsize=7,
              color='#888888')

# Jetson Nano box
jetson_box = FancyBboxPatch((0.3, 3.5), 2.2, 3.5,
                             boxstyle="round,pad=0.1",
                             facecolor='#1a0a2a', edgecolor=PURPLE, linewidth=2)
ax_scene.add_patch(jetson_box)
ax_scene.text(1.4, 6.5, '🖥️', ha='center', fontsize=14)
ax_scene.text(1.4, 5.8, 'JETSON', ha='center', fontsize=7,
              color=PURPLE, fontweight='bold')
ax_scene.text(1.4, 5.3, 'NANO', ha='center', fontsize=7,
              color=PURPLE, fontweight='bold')
ax_scene.text(1.4, 4.7, 'CNN Model', ha='center', fontsize=7, color='#888888')
ax_scene.text(1.4, 4.2, 'Processing...', ha='center', fontsize=7, color='#444444')

# Alert box (bottom)
alert_box = FancyBboxPatch((0.3, 0.1), 9.0, 1.8,
                            boxstyle="round,pad=0.1",
                            facecolor='#1a0000', edgecolor=RED, linewidth=2)
ax_scene.add_patch(alert_box)
alert_text = ax_scene.text(4.8, 1.0, '⏸  SYSTEM READY — PRESS PLAY',
                            ha='center', fontsize=9, color='#666666', fontweight='bold')

# ── ANIMATED WAVE LINES (radar waves) ──
wave_lines = []
for _ in range(5):
    line, = ax_scene.plot([], [], color=BLUE, linewidth=1.2, alpha=0)
    wave_lines.append(line)

# Reflection lines
ref_lines = []
for _ in range(5):
    line, = ax_scene.plot([], [], color=TEAL, linewidth=1.0, alpha=0, linestyle='--')
    ref_lines.append(line)

# Heartbeat dots on human
heart_dot = ax_scene.plot(8.25, 6.0, 'o', color=RED, markersize=0)[0]

# ── LIVE PLOT LINES ──
raw_line,    = ax_raw.plot([], [],    color=CORAL, linewidth=1.0)
fft_line_h,  = ax_fft.plot([], [],   color=BLUE,  linewidth=1.2, label='Human')
fft_line_e,  = ax_fft.plot([], [],   color='#444', linewidth=1.0, label='Empty', linestyle='--')
breath_line, = ax_breath.plot([], [], color=TEAL,  linewidth=1.2, label='Breathing')
heart_line,  = ax_breath.plot([], [], color=RED,   linewidth=1.0, label='Heartbeat')
conf_line,   = ax_conf.plot([], [],   color=AMBER, linewidth=1.5)
conf_fill    = ax_conf.fill_between([], [], alpha=0.2, color=AMBER)

ax_fft.legend(facecolor='#1a1a1a', edgecolor='#333333',
              labelcolor='white', fontsize=7)
ax_breath.legend(facecolor='#1a1a1a', edgecolor='#333333',
                 labelcolor='white', fontsize=7)
ax_fft.set_xlim(0, 3.5)
ax_conf.set_xlim(0, 100); ax_conf.set_ylim(0, 110)
ax_conf.axhline(50, color='white', linewidth=1, linestyle='--', alpha=0.5)
ax_conf.text(2, 52, 'Threshold', color='#888888', fontsize=7)

# Vital signs monitor elements
ax_vitals.set_xlim(0, 1); ax_vitals.set_ylim(0, 1)
ax_vitals.axis('off')
breath_bar_bg = FancyBboxPatch((0.05, 0.62), 0.9, 0.14,
                                boxstyle="round,pad=0.01",
                                facecolor='#1a1a1a', edgecolor='#333333', linewidth=1)
heart_bar_bg  = FancyBboxPatch((0.05, 0.32), 0.9, 0.14,
                                boxstyle="round,pad=0.01",
                                facecolor='#1a1a1a', edgecolor='#333333', linewidth=1)
ax_vitals.add_patch(breath_bar_bg)
ax_vitals.add_patch(heart_bar_bg)
ax_vitals.text(0.5, 0.92, 'VITAL SIGNS MONITOR', ha='center', fontsize=8,
               color='white', fontweight='bold')
ax_vitals.text(0.05, 0.80, '🫁 Breathing Rate:', fontsize=8, color=TEAL)
ax_vitals.text(0.05, 0.50, '❤️  Heart Rate:', fontsize=8, color=RED)
breath_bar = FancyBboxPatch((0.05, 0.62), 0.0, 0.14,
                             boxstyle="round,pad=0.01",
                             facecolor=TEAL, edgecolor='none')
heart_bar  = FancyBboxPatch((0.05, 0.32), 0.0, 0.14,
                             boxstyle="round,pad=0.01",
                             facecolor=RED,  edgecolor='none')
ax_vitals.add_patch(breath_bar)
ax_vitals.add_patch(heart_bar)
breath_val_text = ax_vitals.text(0.5, 0.69, '-- /min', ha='center',
                                  fontsize=8, color='white', fontweight='bold')
heart_val_text  = ax_vitals.text(0.5, 0.39, '-- bpm',  ha='center',
                                  fontsize=8, color='white', fontweight='bold')
ax_vitals.text(0.5, 0.15, 'Status:', ha='center', fontsize=8, color='#888888')
vitals_status = ax_vitals.text(0.5, 0.05, 'SCANNING...', ha='center',
                                fontsize=9, color='#666666', fontweight='bold')

# ============================================================
# DATA BUFFERS
# ============================================================
t_counter     = [0]
raw_buf       = []
fft_buf_h     = []
breath_buf    = []
heart_buf     = []
conf_history  = []
frame_counter = [0]

# Phase control — alternate human/empty every 5 seconds
PHASE_DURATION = 500  # frames
is_human       = [True]

# ============================================================
# ANIMATION UPDATE
# ============================================================
def update(frame):
    t = frame / FS
    t_counter[0] = t
    frame_counter[0] += 1

    # Switch phase every PHASE_DURATION frames
    phase_frame = frame_counter[0] % (PHASE_DURATION * 2)
    is_human[0] = phase_frame < PHASE_DURATION

    # Get new sample
    if is_human[0]:
        sample = get_human_sample(t)
    else:
        sample = get_empty_sample(t)

    raw_buf.append(sample)
    if len(raw_buf) > BUFFER_SIZE:
        raw_buf.pop(0)

    # ── Raw signal plot ──
    raw_arr = np.array(raw_buf)
    raw_line.set_data(range(len(raw_arr)), raw_arr)
    ax_raw.set_xlim(0, BUFFER_SIZE)
    ax_raw.set_ylim(-2.5, 2.5)

    color = CORAL if is_human[0] else '#555555'
    raw_line.set_color(color)

    # ── FFT ──
    if len(raw_buf) >= 64:
        fft_vals = np.abs(np.fft.rfft(raw_arr, n=256))
        freqs    = np.fft.rfftfreq(256, 1/FS)
        mask     = freqs < 3.5
        fft_line_h.set_data(freqs[mask], fft_vals[mask])
        ax_fft.set_ylim(0, max(fft_vals[mask]) * 1.2 + 1)

        # Breathing + heartbeat
        if len(raw_buf) >= 20:
            filtered = bandpass_filter(raw_arr, 0.1, 2.5)
            breath_f = bandpass_filter(raw_arr, 0.1, 0.5)
            heart_f  = bandpass_filter(raw_arr, 0.8, 2.5)
            breath_line.set_data(range(len(breath_f)), breath_f)
            heart_line.set_data(range(len(heart_f)),   heart_f)
            ax_breath.set_xlim(0, BUFFER_SIZE)
            ax_breath.set_ylim(-1.2, 1.2)

    # ── Confidence ──
    conf = fake_cnn_confidence(raw_buf, is_human[0])
    conf_history.append(conf)
    if len(conf_history) > 100:
        conf_history.pop(0)

    conf_arr = np.array(conf_history)
    conf_line.set_data(range(len(conf_arr)), conf_arr)

    # Update confidence fill
    for coll in ax_conf.collections:
        coll.remove()
    ax_conf.fill_between(range(len(conf_arr)), conf_arr, 50,
                          where=conf_arr >= 50, alpha=0.25, color=RED,
                          interpolate=True)
    ax_conf.fill_between(range(len(conf_arr)), conf_arr, 50,
                          where=conf_arr < 50, alpha=0.25, color=TEAL,
                          interpolate=True)
    ax_conf.set_xlim(0, max(100, len(conf_arr)))

    # ── Vital signs bars ──
    if is_human[0]:
        br_w = min(0.9, 0.5 + 0.4 * abs(np.sin(2*np.pi*BREATH_RATE*t)))
        hb_w = min(0.9, 0.5 + 0.4 * abs(np.sin(2*np.pi*HEART_RATE*t)))
        breath_bar.set_width(br_w)
        heart_bar.set_width(hb_w)
        breath_val_text.set_text(f'{BREATH_RATE*60:.0f} /min')
        heart_val_text.set_text(f'{HEART_RATE*60:.0f} bpm')
        vitals_status.set_text('⚠️  VITAL SIGNS DETECTED')
        vitals_status.set_color(RED)
    else:
        breath_bar.set_width(0.05)
        heart_bar.set_width(0.05)
        breath_val_text.set_text('-- /min')
        heart_val_text.set_text('-- bpm')
        vitals_status.set_text('✅  AREA CLEAR')
        vitals_status.set_color(GREEN)

    # ── Radar waves animation ──
    wave_phase = (frame % 30) / 30.0
    for i, wl in enumerate(wave_lines):
        offset = i / len(wave_lines)
        phase  = (wave_phase + offset) % 1.0
        x_pos  = 2.5 + phase * 4.0
        if x_pos < 6.5:
            wy = np.linspace(7, 12, 20)
            wx = x_pos + 0.3 * np.sin(wy * 2)
            wl.set_data(wx, wy)
            wl.set_alpha(max(0, 0.8 - phase))
            wl.set_color(BLUE)
        else:
            wl.set_data([], [])

    # Reflection waves (only when human)
    for i, rl in enumerate(ref_lines):
        if is_human[0]:
            offset = i / len(ref_lines)
            phase  = (wave_phase + offset) % 1.0
            x_pos  = 7.0 - phase * 4.5
            if x_pos > 2.5:
                ry = np.linspace(6, 11, 20)
                rx = x_pos + 0.2 * np.sin(ry * 2 + 1)
                rl.set_data(rx, ry)
                rl.set_alpha(max(0, 0.6 - phase * 0.5))
                rl.set_color(TEAL)
            else:
                rl.set_data([], [])
        else:
            rl.set_data([], [])

    # Heartbeat pulse dot
    if is_human[0]:
        pulse = abs(np.sin(2 * np.pi * HEART_RATE * t))
        heart_dot.set_markersize(pulse * 18)
        heart_dot.set_color(RED)
        heart_dot.set_alpha(pulse)
    else:
        heart_dot.set_markersize(0)

    # ── Detection Status Panel ──
    ax_status.clear()
    ax_status.set_facecolor('#0f0f0f')
    ax_status.axis('off')
    ax_status.set_xlim(0, 1); ax_status.set_ylim(0, 1)

    if is_human[0]:
        bg_col  = '#1a0000'
        brd_col = RED
        icon    = '🚨'
        txt     = 'HUMAN\nDETECTED'
        txt_col = RED
        sub     = '⚠️  ALERT RESCUE TEAM!'
        sub_col = '#FF8888'
        fig.patch.set_facecolor('#100000')
    else:
        bg_col  = '#001000'
        brd_col = GREEN
        icon    = '✅'
        txt     = 'AREA\nCLEAR'
        txt_col = GREEN
        sub     = 'No human detected'
        sub_col = '#88FF88'
        fig.patch.set_facecolor('#080808')

    ax_status.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                        boxstyle="round,pad=0.02",
                                        facecolor=bg_col, edgecolor=brd_col,
                                        linewidth=3))
    ax_status.text(0.5, 0.88, icon,    ha='center', fontsize=28)
    ax_status.text(0.5, 0.65, txt,     ha='center', fontsize=16,
                   color=txt_col, fontweight='bold', va='center')
    ax_status.text(0.5, 0.44, sub,     ha='center', fontsize=9,
                   color=sub_col, va='center')
    ax_status.text(0.5, 0.30, f'Confidence: {conf:.1f}%', ha='center',
                   fontsize=11, color='white', fontweight='bold')

    # Confidence mini bar
    ax_status.add_patch(FancyBboxPatch((0.08, 0.18), 0.84, 0.08,
                                        boxstyle="round,pad=0.01",
                                        facecolor='#1a1a1a', edgecolor='#333333'))
    bar_w = 0.84 * conf / 100
    ax_status.add_patch(FancyBboxPatch((0.08, 0.18), bar_w, 0.08,
                                        boxstyle="round,pad=0.01",
                                        facecolor=brd_col, edgecolor='none'))

    # Phase timer
    frames_left = PHASE_DURATION - (phase_frame if is_human[0] else phase_frame - PHASE_DURATION)
    secs_left   = max(0, frames_left / 10)
    ax_status.text(0.5, 0.08, f'Phase: {"Human" if is_human[0] else "Empty"} | Next: {secs_left:.0f}s',
                   ha='center', fontsize=7, color='#555555')

    # ── Alert box in scene ──
    alert_text.set_text(
        '🚨  HUMAN DETECTED — ALERT RESCUE TEAM!' if is_human[0]
        else '✅  AREA CLEAR — NO HUMAN DETECTED'
    )
    alert_text.set_color(RED if is_human[0] else GREEN)

    return ([raw_line, fft_line_h, breath_line, heart_line,
             conf_line, heart_dot, alert_text, breath_bar,
             heart_bar, breath_val_text, heart_val_text, vitals_status]
            + wave_lines + ref_lines)

# ============================================================
# RUN ANIMATION
# ============================================================
print("="*60)
print("🚀 STARTING ANIMATED RADAR SIMULATION")
print("="*60)
print("  🟢 First 5 seconds  → Human detected (red alert)")
print("  🔵 Next  5 seconds  → Area clear (green)")
print("  🔄 Repeats continuously")
print("  ❌ Close window to stop")
print("="*60)

ani = animation.FuncAnimation(
    fig, update,
    frames=2000,
    interval=100,
    blit=False,
    cache_frame_data=False
)

plt.show()
