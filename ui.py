import importlib
from outsmart import return_copy
import outsmart as osmt
import pyglet
import glob


############################################
# End of game
############################################
@return_copy
def victory(s):
    """Print victory, let the option to leave to main menu"""
    s.ui.end_text = """VICTORY !"""
    s.ui.story_text = ""
    s.ui.obj_text = ""
    s.ui.log_text = ""
    s.ui.active = ALL_INACTIVE.copy()
    s.ui.active["lab_wild_quit"] = True
    return s


@return_copy
def defeat(s):
    """Print defeat, play song, let the option of trying again"""
    s.ui.end_text = "Defeat !\n"
    "The robots grew in number and wiped out "
    "the human race."
    s.ui.story_text = ""
    s.ui.obj_text = ""
    s.ui.log_text = ""
    s.ui.active = ALL_INACTIVE.copy()
    s.ui.active["retry"] = True
    s.ui.active["lab_wild_quit"] = True
    return s


############################################
# Modes
############################################
def import_lvl(name):
    # https://stackoverflow.com/questions/27381264/python-3-4-how-to-import-a-module-given-the-full-path
    fname = glob.glob(name+'/*.py')[0]
    importlib.machinery.SourceFileLoader(name, fname).load_module()


def level_buttons():
    lvl_directory = "levels"  # DEFAULT
    y_offset = 0
    answer = {}
    for dir in glob.glob(lvl_directory+"/*"):
        img = pyglet.image.load(dir+"/img.png")
        name = dir.split('/')[1][2:]  # FIXME use os.path.split()
        print("Dir : "+dir+", name :"+name)
        x = osmt.WINDOW.width//2-img.width//2
        y = osmt.WINDOW.height-img.height - y_offset
        y_offset += img.height+20
        answer["main_"+name] = [x, y, pyglet.image.load(dir+"/img.png"),
                                lambda dir=dir: import_lvl(dir)]
    return answer

ALL_INACTIVE = {k: False for k in ["lab_wild_reset", "lab_train",
                                   "editor_load", "lab_step",
                                   "editor_save", "lab_go_wild",
                                   "wild_go_lab", "lab_wild_quit", "retry",
                                   "lab_prev_tile", "lab_next_tile",
                                   "lab_current_tile", "lab_wild_step",
                                   "lab_right",
                                   "lab_left", "lab_up", "lab_down",
                                   "lab_pick"]+list(level_buttons().keys())}

LAB_ACTIVE = ALL_INACTIVE.copy()
LAB_ACTIVE.update({k: True for k in ALL_INACTIVE if "lab_" in k})

WILD_ACTIVE = ALL_INACTIVE.copy()
WILD_ACTIVE.update({k: True for k in ALL_INACTIVE if "wild_" in k})

EDITOR_ACTIVE = LAB_ACTIVE.copy()
WILD_ACTIVE.update({k: True for k in ALL_INACTIVE if "editor_" in k})

MAIN_MENU_ACTIVE = ALL_INACTIVE.copy()
MAIN_MENU_ACTIVE.update({k: True for k in level_buttons()})


############################################
# Events
############################################
@return_copy
def reset(s):
    """Reset the Q-function of the robot"""
    s = osmt.reset(s)
    s.log_text = "Bob's learning has been reset."
    return s


@return_copy
def train(s):
    """Train the robot, print the result"""
    try:
        s.rl = osmt.train(s.lab, s.rl)
    except AssertionError:
        s.log_text = "Error ! No reward found this time."
    else:
        s.log_text = "Training succeeded."
    return s


@return_copy
def step(s):
    """Make a step in the current matrix"""
    s = s.ui.walk(s, 1)
    return s


@return_copy
def load(s):
    """Load the lab and wild matrix from file"""
    filename = s.ui.filename
    s = osmt.load_state(s, filename)
    s.log_text = "Matrices loaded"
    return s


@return_copy
def save(s):
    """Save the lab and wild matrices"""
    filename = s.ui.filename
    osmt.save_state(s, filename)
    s.log_text = "Matrices saved"
    return s


@return_copy
def wild(s):
    """Switch the view to the wild"""
    s.ui.terrain = get_wild
    s.ui.set_terrain = set_wild
    s.ui.walk = osmt.walk_wild
    s.ui.active = WILD_ACTIVE.copy()
    s.log_text = "Now into the wild."
    return s


@return_copy
def lab(s):
    """Switch the view to the lab"""
    s.ui.terrain = get_lab
    s.ui.set_terrain = set_lab
    s.ui.walk = osmt.walk_wild
    s.ui.active = LAB_ACTIVE.copy()
    s.log_text = "Now back to the lab."
    return s


@return_copy
def _quit(s):
    """Go to main menu"""
    s.ui.active = MAIN_MENU_ACTIVE.copy()
    return s


CHECKPOINT = None


@return_copy
def retry(s):
    """Load last checkpoint"""
    return osmt.copy(CHECKPOINT)


def checkpoint_now(s):
    """Save current state"""
    global CHECKPOINT
    CHECKPOINT = osmt.copy(s)


GALLERY = [100, 110, 200, 210, 300, 310, 400, 410]


@return_copy
def prev_tile(s):
    """Select prev tile in the tile tool"""
    s.ui._current_tile = GALLERY.index(s.ui.current_tile) - 1 % len(GALLERY)
    return s


@return_copy
def next_tile(s):
    """Select next tile in the tile tool"""
    s.ui._current_tile = GALLERY.index(s.ui.current_tile) + 1 % len(GALLERY)
    return s


@return_copy
def tile_tool(s):
    """Toggle activation of the tile tool"""
    s.ui.tile_tool = not s.ui.tile_tool
    return s


def current_tile(s):
    """Return the image for the current tile"""
    return GALLERY[s.ui._current_tile]


@return_copy
def apply_action(s, a):
    """Apply the given action to the relevant matrix"""
    s.ui.set_terrain(s, osmt.apply_action(s.ui.terrain(), a))
    return s


@return_copy
def cursor_at(s, i, j):
    """Save cursor position for futur temp modification of terrain"""
    s.ui.cursor = [i, j]
    return s


@return_copy
def cursor_out(s):
    """Take note that cursor is out of the terrain"""
    s.ui.cursor = False
    return s


@return_copy
def click(s, i, j):
    """Validate terrain modification if appropriate"""
    if s.ui.terrain == get_wild:
        return
    m = s.ui.terrain().copy()
    m[i, j] = GALLERY[s.ui.current_tile]
    s.ui.set_terrain(s, m)
    return s


@return_copy
def set_lab(s, m):
    s.lab = m.copy()
    return s


def get_lab(s):
    answer = s.lab.copy()
    if s.ui.cursor:
        answer[tuple(s.ui.cursor)] = GALLERY[s.ui.current_tile]
    return answer


@return_copy
def set_wild(s, m):
    s.wild = m.copy()
    return s


def get_wild(s):
    return s.wild.copy()


class UI():
    def __init__(self):
        self.obj_text = ""  # Displayed in the upper right
        self.story_text = ""  # Displayed in the upper left
        self.log_text = ""  # Displayed in the lower left
        self.end_text = ""  # Center of screen, big

        self.active = ALL_INACTIVE.copy()  # Active ui elements

        self.current_tile = 0  # For the terrain editor
        self.cursor = False  # Tile the user is hovering

        self.defeat = defeat
        self.victory = victory
        self.terrain = get_lab
        self.set_terrain = set_lab
        self.walk = osmt.walk_lab

        self.filename = "levels/99level_editor/level_editor"  # DEFAULT
