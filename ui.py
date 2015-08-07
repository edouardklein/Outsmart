from outsmart import return_copy
import outsmart as osmt


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
    s.ui.active["end_text"] = True
    return s


############################################
# Modes
############################################

ALL_INACTIVE = {k: False for k in ["lab_wild_reset", "lab_train",
                                   "editor_load", "editor_save", "lab_go_wild",
                                   "wild_go_lab", "lab_wild_quit", "retry",
                                   "lab_prev_tile", "lab_next_tile",
                                   "lab_current_tile", "lab_wild_step",
                                   "lab_right",
                                   "lab_left", "lab_up", "lab_down",
                                   "lab_pick", "story_text", "obj_text",
                                   "log_text", "end_text",
                                   "editor_wild_lab_terrain"]}

LAB_ACTIVE = ALL_INACTIVE.copy()
LAB_ACTIVE.update({k: True for k in ALL_INACTIVE if "lab_" in k})
LAB_ACTIVE.update({k: True for k in ALL_INACTIVE if "_text" in k})

WILD_ACTIVE = ALL_INACTIVE.copy()
WILD_ACTIVE.update({k: True for k in ALL_INACTIVE if "wild_" in k})
WILD_ACTIVE.update({k: True for k in ALL_INACTIVE if "_text" in k})

EDITOR_ACTIVE = LAB_ACTIVE.copy()
EDITOR_ACTIVE.update({k: True for k in ALL_INACTIVE if "editor_" in k})


############################################
# Events
############################################
@return_copy
def reset(s):
    """Reset the Q-function of bob or the position of the red robot"""
    if s.ui.terrain == get_lab:
        s = osmt.reset(s)
        s.log_text = "Bob's learning has been reset."
    else:  # Wild
        s.wild = CHECKPOINT.wild.copy()
    return s


@return_copy
def train(s):
    """Train the robot, print the result"""
    try:
        s.rl = osmt.train(s.lab, s.rl)
    except AssertionError:
        s.ui.log_text = "Error ! No reward found this time."
    else:
        s.ui.log_text = "Training succeeded."
    return s


@return_copy
def step(s):
    """Make a step in the current matrix"""
    s = s.ui.walk(s, 1)
    return s


@return_copy
def load(s):
    """Load the lab and wild matrix from file"""
    s = osmt.load_state(s, s.ui.filename)
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
    s.ui.walk = osmt.walk_lab
    s.ui.active = LAB_ACTIVE.copy()
    s.log_text = "Now back to the lab."
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
    i = (GALLERY.index(s.ui.current_tile) - 1) % len(GALLERY)
    s.ui.current_tile = GALLERY[i]
    return s


@return_copy
def next_tile(s):
    """Select next tile in the tile tool"""
    i = (GALLERY.index(s.ui.current_tile) + 1) % len(GALLERY)
    s.ui.current_tile = GALLERY[i]
    return s


@return_copy
def tile_tool(s):
    """Toggle activation of the tile tool"""
    s.ui.tile_tool = not s.ui.tile_tool
    return s


@return_copy
def apply_action(s, a):
    """Apply the given action to the relevant matrix"""
    s = s.ui.set_terrain(s, osmt.apply_action(s.ui.terrain(s), a))
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
        return s
    m = s.ui.terrain(s).copy()  # Will have tile modification
    # if tile_tool is True, see get_lab()
    if not s.ui.tile_tool:
        m = osmt.move_robot(m, i, j)
    s = s.ui.set_terrain(s, m)
    return s


@return_copy
def set_lab(s, m):
    s.lab = m.copy()
    return s


def get_lab(s):
    answer = s.lab.copy()
    if s.ui.cursor and s.ui.tile_tool:
        c = tuple(s.ui.cursor)
        answer[c] = s.ui.current_tile + answer[c] % 10
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

        self.current_tile = 100  # For the terrain editor
        self.cursor = False  # Tile the user is hovering
        self.tile_tool = False

        self.defeat = defeat
        self.victory = victory
        self.terrain = get_lab
        self.set_terrain = set_lab
        self.walk = osmt.walk_lab

        self.filename = "levels/99level_editor/level_editor"  # DEFAULT
