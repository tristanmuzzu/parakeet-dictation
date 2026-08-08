#!/usr/bin/env bash
# Register Parakeet Dictation's hotkeys as GNOME custom keybindings.
#
#   bash install-gnome-hotkeys.sh          # install / update
#   bash install-gnome-hotkeys.sh --remove # take them back out
#
# Why GNOME owns the keys: under Wayland no application can grab global
# hotkeys or read keys aimed at another window. The compositor is the only
# thing allowed to, so we ask it to run `dictation.py --send <cmd>`, which
# pokes the already-running app over its loopback command channel.
#
# NOTE ON THE BINDING: the Windows build toggles on a bare Ctrl+Win chord.
# GNOME keybindings must include a non-modifier key, so a bare two-modifier
# chord cannot be expressed. Defaults below are the nearest comfortable
# equivalents; override with env vars, e.g.
#   TOGGLE_KEY='<Super>d' bash install-gnome-hotkeys.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3)"

TOGGLE_KEY="${TOGGLE_KEY:-<Control><Super>space}"
CONT_KEY="${CONT_KEY:-<Control><Shift><Super>space}"
CANCEL_KEY="${CANCEL_KEY:-<Control><Super>Escape}"
QUIT_KEY="${QUIT_KEY:-<Control><Alt>q}"

ROOT="org.gnome.settings-daemon.plugins.media-keys"
BASE="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings"
PREFIX="parakeet"

# Existing custom bindings, minus any of ours (so re-running is idempotent).
# An empty list comes back as `@as []` - that type annotation must not survive
# into the new list, or GNOME gets a bogus '@as' entry it will never resolve.
mapfile -t KEEP < <(
  gsettings get "$ROOT" custom-keybindings \
    | tr -d "[]' " | tr ',' '\n' \
    | grep -v "^$" | grep -v '^@as$' | grep '^/' | grep -v "/${PREFIX}-" || true
)

if [[ "${1:-}" == "--remove" ]]; then
  for i in toggle continuous cancel quit; do
    dconf reset -f "$BASE/${PREFIX}-${i}/" 2>/dev/null || true
  done
  if [[ ${#KEEP[@]} -eq 0 ]]; then
    gsettings set "$ROOT" custom-keybindings "@as []"
  else
    printf -v joined "'%s'," "${KEEP[@]}"
    gsettings set "$ROOT" custom-keybindings "[${joined%,}]"
  fi
  echo "removed parakeet keybindings"
  exit 0
fi

# A GNOME/mutter binding always beats a custom one, and the custom one then
# fails *silently* - which is exactly how <Super>d (show-desktop) and
# <Super>space (input-source switch) wasted an afternoon. Say so up front.
check_conflict() {
  local key="$1" owner
  owner=$(gsettings list-recursively 2>/dev/null \
          | grep -F "'${key}'" | grep -v "custom-keybinding" | head -1)
  [[ -n "$owner" ]] && printf "  !! %s is already bound: %s\n     GNOME wins; this shortcut will do nothing.\n" "$key" "${owner%% *}"
}

add() {  # name  binding  command  slug
  local slug="$4" path="$BASE/${PREFIX}-$4/"
  check_conflict "$2"
  gsettings set "${ROOT}.custom-keybinding:$path" name    "$1"
  gsettings set "${ROOT}.custom-keybinding:$path" binding "$2"
  gsettings set "${ROOT}.custom-keybinding:$path" command "$3"
  KEEP+=("$path")
  printf "  %-26s %s\n" "$2" "$1"
}

echo "installing GNOME keybindings:"
add "Parakeet: toggle dictation"   "$TOGGLE_KEY" "$PY $REPO/dictation.py --send toggle"            toggle
add "Parakeet: continuous mode"    "$CONT_KEY"   "$PY $REPO/dictation.py --send toggle-continuous" continuous
add "Parakeet: cancel recording"   "$CANCEL_KEY" "$PY $REPO/dictation.py --send cancel"            cancel
add "Parakeet: quit"               "$QUIT_KEY"   "$PY $REPO/dictation.py --send quit"              quit

printf -v joined "'%s'," "${KEEP[@]}"
gsettings set "$ROOT" custom-keybindings "[${joined%,}]"

echo
echo "done. Verify under Settings > Keyboard > View and Customize Shortcuts > Custom."
echo "If a binding does nothing, it is almost always already taken by GNOME -"
echo "the Settings panel shows the conflict; pick another and re-run with e.g."
echo "  TOGGLE_KEY='<Super>d' bash install-gnome-hotkeys.sh"
