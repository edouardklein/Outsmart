#!/usr/bin/env python3
import graphics
import ui


graphics.STATE.ui.active = ui.ALL_INACTIVE
graphics.STATE.ui.active.update(ui.EDITOR_ACTIVE)
