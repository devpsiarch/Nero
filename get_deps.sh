#!/bin/bash

set -e

install_apt() {
  echo "[+] Detected APT-based system"
  sudo apt update
  sudo apt install -y build-essential libc6-dev libraylib-dev
}

install_dnf() {
  echo "[+] Detected DNF-based system"
  sudo dnf groupinstall -y "Development Tools"
  sudo dnf install -y glibc-devel raylib-devel
}

install_pacman() {
  echo "[+] Detected Pacman-based system"
  sudo pacman -Sy --noconfirm base-devel glibc raylib
}

echo "[*] Detecting package manager..."

if command -v apt &> /dev/null; then
  install_apt
elif command -v dnf &> /dev/null; then
  install_dnf
elif command -v pacman &> /dev/null; then
  install_pacman
else
  echo "[-] Unsupported package manager. Please install the dependencies manually."
  exit 1
fi

echo "[+] All dependencies installed successfully."
