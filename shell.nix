{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  name = "TLNet";

  venvDir = "./.venv";

  buildInputs = with pkgs; [
    python39
    python39Packages.pip
    python39Packages.setuptools

    python39Packages.venvShellHook
  ];

}
