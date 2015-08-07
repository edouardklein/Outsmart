import pyglet
from pyglet.window import key
import math
from outsmart import return_copy
import outsmart as osmt
import ui
import glob
import functools
import itertools
import hashlib
import os
import sys
import subprocess
import importlib

STATE = None


def _state(func):
    global STATE
    STATE = func(STATE)


############################################
# Buttons
############################################
def draw_text(text_list, center=False):
    """Draw the given [x, y, size, text] list"""
    anchor_x = 'left' if not center else 'center'
    anchor_y = 'baseline' if not center else 'center'
    for x, y, size, t in text_list:
        label = pyglet.text.Label(t, x=x, y=y,
                                  font_name='KenVector Future Thin',
                                  font_size=size,
                                  anchor_x=anchor_x,
                                  anchor_y=anchor_y)
        label.draw()

BUTTON_LEFT = pyglet.image.load('img/button_left.png')
BUTTON_MID = pyglet.image.load('img/button_mid.png')
BUTTON_RIGHT = pyglet.image.load('img/button_right.png')


def draw_text_button(x, y, text):
    """Draw a button containing some text"""
    sprite = pyglet.sprite.Sprite(BUTTON_LEFT, x=x, y=y)
    sprite.draw()
    pixel_length = len(text)*10.5
    for i in range(0, math.ceil(pixel_length/16)):
        sprite = pyglet.sprite.Sprite(BUTTON_MID, x=x+6+i*16, y=y)
        sprite.draw()
    sprite = pyglet.sprite.Sprite(BUTTON_RIGHT, x=x+pixel_length, y=y)
    sprite.draw()
    draw_text([[x+6, y+9, 12, text]])


def draw_buttons(s):
    """Draw text and images buttons"""
    for [x, y, content, _] in [BUTTONS[k] for k in BUTTONS
                               if s.ui.active[k]]:
        if type(content) == str:
            draw_text_button(x, y, content)
        else:  # image button
            sprite = pyglet.sprite.Sprite(content(s), x=x, y=y)
            sprite.draw()


def check_buttons(s, _x, _y, button):
    if button == pyglet.window.mouse.LEFT:
        for [x, y, content, func] in [BUTTONS[k] for k in BUTTONS
                                      if STATE.ui.active[k]]:
            if type(content) == str:
                max_x = x+len(content)*10.5+10
                max_y = y+26
            else:  # image button
                max_x = x+content(s).width
                max_y = y+content(s).height
            if _x >= x and _y >= y and _x <= max_x and _y <= max_y:
                func()
                return True
    return False


def import_lvl(name):
    # https://stackoverflow.com/questions/27381264/python-3-4-how-to-import-a-module-given-the-full-path
    fname = glob.glob(name+'/*.py')[0]
    importlib.machinery.SourceFileLoader(name, fname).load_module()


def level_buttons():
    lvl_directory = "levels"  # DEFAULT
    y_offset = 0
    answer = {}
    for dirr in glob.glob(lvl_directory+"/*"):
        name = os.path.split(dirr)[1][2:]
        print("Dir : "+dirr+", name :"+name)
        img = pyglet.image.load(dirr+"/img.png")
        x = WINDOW.width//2 - img.width//2
        y = WINDOW.height-img.height - y_offset
        y_offset += img.height+20
        answer["main_"+name] = [x, y, lambda s, img=img: img,
                                lambda dirr=dirr: import_lvl(dirr)]
    return answer


BUTTONS = {"lab_wild_reset": [10, 50, "Reset", lambda: _state(ui.reset)],
           "lab_train": [10, 100, "Train", lambda:  _state(ui.train)],
           "lab_wild_step": [10, 150, "Step", lambda:  _state(ui.step)],
           "editor_load": [10, 200, "Load", lambda:  _state(ui.load)],
           "editor_save": [200, 200, "Save", lambda:  _state(ui.save)],
           "lab_go_wild": [10, 250, "Wild", lambda:  _state(ui.wild)],
           "lab_copy_wild": [100, 250, "Copy wild",
                             lambda:  _state(ui.copy_wild)],
           "wild_go_lab": [200, 250, "Lab", lambda:  _state(ui.lab)],
           "lab_wild_quit": [10, 300, "Exit",
                             lambda: _state(_quit)],

           "retry": [500, 500,
                     lambda s: pyglet.image.load('img/retry.png'),
                     lambda: _state(ui.retry)],

           "lab_prev_tile": [1000, 20,
                             lambda s: pyglet.image.load('img/larrow.png'),
                             lambda: _state(ui.prev_tile)],
           "lab_next_tile": [1100, 20,
                             lambda s: pyglet.image.load('img/rarrow.png'),
                             lambda: _state(ui.next_tile)],
           "lab_current_tile": [1000, 50,
                                lambda s: nb2images(s.ui.current_tile)[0],
                                lambda: _state(ui.tile_tool)]}


############################################
# Tiling
############################################
TILE_SIZE_X = 64
TILE_SIZE_Y = 32

X_MIN = 2*(osmt.I_MAX+1)*TILE_SIZE_X
Y_MIN = (2*(osmt.J_MAX+1)+2)*TILE_SIZE_Y
# The display system :
# Each tile has a multi-digit number associated with it.
# Each position encodes a different information.
# The least significant digit is the presence of the robot:
#  - 0: No robot
#  - 1: Bob
#  - 2: Not-Bob
# The second digit is the presence of a trap:
#  - 0: No trap
#  - 1: Trap
# The third digit from the right is the type of terrain:
# - 0: No terrain (no use case, but I feel good knowing 0 means 'nothing')
# - 1: bare (e.g. Earth)
# - 2: No resources (e.g. grass)
# - 3: Resource type 1 (e.g. Crystals)
# - 4: Resouce type 2 (e.g. Rocks)
# The fourth digit is the set of tiles to use (to be implemented)
# - 0: Earth, grass, crystals, rocks
# - 1: (not implemented yet) e.g. grass, road, house, trees
# - 2: ...
# A tile is a superposition of the information at each position.
# Each position encodes things that can not be superimposed (e.g. it is not
# possible to have both crystals and rocks on the same tiles)
# On the other hand it is possible to have a Not-Bob on a trap on some rocks.
# This would tile 0412
# To draw tile 0412, either the file 0412.png exists and is displayed
# or we build it from files that sum to 0412.png, e.g. 0410 and 0002
IMAGES = {int(s[-8:-4]): pyglet.image.load(s)
          for s in glob.glob('img/'+'[0-9]'*4+'.png')}


@functools.lru_cache(None)
def nb2images(number):
    """Return the list of images one has to draw to represent the
    given number"""
    try:
        return [IMAGES[number]]
    except KeyError:
        all_comb = sum(map(list, [itertools.combinations(IMAGES.keys(), l)
                                  for l in [2, 3, 4]]), [])
        comb = [c for c in all_comb if sum(c) == number][0]
        return [IMAGES[n] for n in reversed(sorted(comb))]


def draw_sprite(number, x, y):
    """Draw at x,y the sprite(s) representing the given number"""
    for im in nb2images(number):
        sprite = pyglet.sprite.Sprite(im, x, y)
        sprite.draw()


def ij2xy(m, i, j):
    """We use matrix-oriented coordinates i,j, but to display we need oriented
    abc/ord x, y instead, using isometric projection"""
    x = (m.shape[0] - i + j - 1)*TILE_SIZE_X
    y = (m.shape[1]+m.shape[0] - i - j - 2)*TILE_SIZE_Y
    return x, y


def xy2ij(x, y):
    """Return the i, j matrix indices after a reverse projection of x, y"""
    ix = x
    iy = ((2*(osmt.J_MAX+1)+2)*TILE_SIZE_Y-y)
    ix = ix / TILE_SIZE_X / 2
    iy = iy / TILE_SIZE_Y / 2 - .2
    i = round(iy-ix)+4
    j = round(ix+iy)-6  # TGCM!
    return i, j


############################################
# Audio
############################################
DEFEAT_SONG = pyglet.media.load('snd/dead.mp3')


def play(text="", media=None, media_file=""):
    if media:
        play_media(media)
    elif media_file:
        play_file(media_file)
    else:  # text
        if text.startswith("Defeat !"):
            play_file('snd/dead.mp3')  # Play defeat song
            return
        play_text(text)


def play_text(text):
    """Play the correct media file, generating it if necessary"""
    media_file = "snd/TTS/%s.aiff" % hashlib.sha1(text.encode()).hexdigest()
    if not os.path.exists(media_file):
        text_to_aiff(text, media_file)
    play_file(media_file)


def text_to_aiff(text, fname):
    """Generate the aiff audio from the given text."""
    print("WARN: TTS file NOT found: %s" % fname)
    if sys.platform.startswith("linux"):
        tts_linux(text, fname)
    elif sys.platform == "darwin":
        tts_mac(text, fname)
    else:
        print("ERR: Platform not recognized, no TTS available")


def tts_linux(text, fname):
    """Create aiff file from text on linux"""
    TTS_exe = "/usr/bin/text2wave"
    TTS_command = "%s -otype aiff -o {out}" % TTS_exe
    cmd = TTS_command.format(out=fname)
    cmd = cmd.split()
    p_cmd = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    p_cmd.communicate(input=text.encode())
    if p_cmd.returncode == 0:
        print("INFO: sound file generated")


def tts_mac(text, fname):
    """Create aiff file from text on mac"""
    TTS_exe = "/usr/bin/say"
    STATE.TTS_command = """%s -v 'Vicki' -o {out} {text}""" % TTS_exe
    text = '"'+text.replace("\n", " ").replace('"', '\"')+'"'
    cmd = STATE.TTS_command.format(out=fname, text=text)
    os.system(cmd)
    print("INFO: sound file generated (maybe)")


def play_file(fname):
    """Play the sound from the given file"""
    play_media(pyglet.media.load(fname))


def play_media(media):
    """Actually play the media, cut the playing one off if necessary"""
    global PLAYER
    PLAYER.queue(media)
    if not PLAYER.playing:
        PLAYER.play()
    else:
        PLAYER.next_source()


############################################
# Text
############################################
pyglet.font.add_file('img/kenvector_future_thin.ttf')
KenFuture = pyglet.font.load('KenVector Future Thin Regular')


def xy_text(x, y, text, size=12):
    """Return the list of [x, y, size, text] items that draw_text() will
    understand"""
    return [[x, y - i*15, size, line] for i, line in enumerate(text)]


OLD_STORY_TEXT = ""


def draw_story_text(text):
    """Display in the upper left corner"""
    global OLD_STORY_TEXT
    draw_text(xy_text(0, Y_MIN-10, text.split("\n")))
    if OLD_STORY_TEXT != text:
        play(text=text)
        OLD_STORY_TEXT = text


def draw_objective_text(text):
    """Display in the upper right corner"""
    draw_text(xy_text(X_MIN-600, Y_MIN-10,
                      ["OBJECTIVES"]+text.split("\n")))


def draw_log_text(text):
    """Display in the lower left corner"""
    draw_text(xy_text(10, 10, text.split("\n")))


def draw_end_text(text):
    """Display over the whole screen"""
    l = text.split("\n")
    draw_text([[WINDOW.width//2, WINDOW.height//2, 36, l[0]]],
              center=True)
    draw_text(xy_text(500, 500, l[1:]))


def draw_all_texts(s):
    draw_story_text(s.ui.story_text) if s.ui.active["story_text"] else None
    draw_objective_text(s.ui.obj_text) if s.ui.active["obj_text"] else None
    draw_log_text(s.ui.log_text) if s.ui.active["log_text"] else None
    draw_end_text(s.ui.end_text) if s.ui.active["end_text"] else None


##################################N##########
# Events
############################################
WINDOW = pyglet.window.Window(fullscreen=True)
if WINDOW.width < X_MIN or WINDOW.height < Y_MIN:
    exit(1)  # Screen too small


def draw_assets(s):
    "Draw the game state"
    if s.ui.active["editor_wild_lab_terrain"]:
        m = s.ui.terrain(s)
        for i in range(0, m.shape[0]):
            for j in range(0, m.shape[1]):
                x, y = ij2xy(m, i, j)
                draw_sprite(m[i, j], x, y)
    draw_all_texts(s)
    draw_buttons(s)


@WINDOW.event
def on_draw():
    global STATE
    WINDOW.clear()
    t = PLAYER.get_texture()
    if t:
        t.blit(0, 0)
    if STATE.victorious(STATE):
        STATE = STATE.ui.victory(STATE)
    elif STATE.losing(STATE):
        STATE = STATE.ui.defeat(STATE)
    elif STATE.obj_func(STATE):
        STATE = STATE.next_func(STATE)
    draw_assets(STATE)


@WINDOW.event
def on_mouse_motion(x, y, dx, dy):
    global STATE
    i, j = xy2ij(x, y)
    if i in range(osmt.I_MAX+1) and j in range(osmt.J_MAX+1):
        STATE = ui.cursor_at(STATE, i, j)
    else:
        STATE = ui.cursor_out(STATE)


BINDINGS = {key.S: ["lab_wild_step", lambda: _state(ui.step)],
            key.Q: ["lab_wild_quit", lambda: _state(_quit)],
            key.W: ["lab_go_wild", lambda: _state(ui.wild)],
            key.L: ["wild_go_lab", lambda: _state(ui.lab)],
            key.T: ["lab_train", lambda: _state(ui.train)],
            key.R: ["lab_wild_reset", lambda: _state(ui.reset)],
            key.RIGHT: ["lab_right",
                        lambda: _state(lambda s: ui.apply_action(s, "RIGHT"))],
            key.LEFT: ["lab_left",
                       lambda: _state(lambda s: ui.apply_action(s, "LEFT"))],
            key.DOWN: ["lab_down",
                       lambda: _state(lambda s: ui.apply_action(s, "DOWN"))],
            key.UP: ["lab_up",
                     lambda: _state(lambda s: ui.apply_action(s, "UP"))],
            key.SPACE: ["lab_pick",
                        lambda: _state(lambda s: ui.apply_action(s, "PICK"))],
            key.D: ["lab_randomize", lambda: _state(ui.randomize)]}


@WINDOW.event
def on_key_press(symbol, modifiers):
    """Call the appropriate binding"""
    try:
        name, func = BINDINGS[symbol]
    except KeyError:
        print("Key not bound, symbol "+str(symbol))
        print(list(BINDINGS.keys()))
        print("e.g. UP is supposed to be : "+str(key.UP))
        return
    if STATE.ui.active[name]:
        func()


@WINDOW.event
def on_mouse_press(x, y, button, modifiers):
    """Check for button then pass the click to ui"""
    global STATE
    if check_buttons(STATE, x, y, button):
        return
    i, j = xy2ij(x, y)
    if i not in range(osmt.I_MAX+1) or j not in range(osmt.J_MAX+1):
        return
    if button == pyglet.window.mouse.LEFT:
        STATE = ui.click(STATE, i, j)  # FIXME handle multiple args in _state


############################################
# Controller code
############################################
@return_copy
def _quit(s):
    """Go to main menu"""
    global PLAYER
    s.ui.active = MAIN_MENU_ACTIVE.copy()
    s.victorious = lambda s: False
    PLAYER.next_source()
    return s


def new_state():
    global STATE
    global PLAYER
    PLAYER = pyglet.media.Player()
    STATE = osmt.State()
    STATE.ui = ui.UI()


new_state()
# Main menu hack
_d = level_buttons()
BUTTONS.update(_d)
MAIN_MENU_ACTIVE = ui.ALL_INACTIVE.copy()
MAIN_MENU_ACTIVE.update({k: True for k in _d})
ui.ALL_INACTIVE = {k: False for k in MAIN_MENU_ACTIVE}
ui.WILD_ACTIVE.update({k: False for k in _d})
ui.LAB_ACTIVE.update({k: False for k in _d})
STATE.ui.active = MAIN_MENU_ACTIVE.copy()
