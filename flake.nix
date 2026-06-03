{
  description = "tokenizers-zig — Zig-native HF-compatible tokenizers (Zig 0.16)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Zig 0.16.0 — nixos-25.05 caps at 0.14.1, so we fetch the official
        # upstream tarball and wrap it as a derivation. Per-system sha256s
        # were captured with `nix-prefetch-url` against ziglang.org on
        # 2026-06-02. The official binaries are the same artifacts shipped
        # to all 0.16.0 users; verifying the hash is the integrity contract.
        zigSources = {
          "x86_64-linux" = {
            url = "https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz";
            sha256 = "001xp99bx6nlidc5142h89fkkxipykxzvwz6a65v8x23lxj9dr3h";
          };
          "aarch64-linux" = {
            url = "https://ziglang.org/download/0.16.0/zig-aarch64-linux-0.16.0.tar.xz";
            sha256 = "05sgaghki2x4ydlizlh8mcbnxd5nxxisnmxcrv3gdiifnazhjjza";
          };
          "x86_64-darwin" = {
            url = "https://ziglang.org/download/0.16.0/zig-x86_64-macos-0.16.0.tar.xz";
            sha256 = "19rbnmvjkx92ql0n71w1c2kxpb9vjn8q6b40w6iccyw7s5z5b1q3";
          };
          "aarch64-darwin" = {
            url = "https://ziglang.org/download/0.16.0/zig-aarch64-macos-0.16.0.tar.xz";
            sha256 = "128l806lalp1gmscrjbgrx5fhlxagrpk3lvf90nmr6w7mbg70gdj";
          };
        };

        zigSrc = zigSources.${system} or null;

        zig_0_16_0 =
          if zigSrc == null then
            throw "tokenizers-zig flake: no pinned Zig 0.16.0 tarball for system '${system}'. Add the sha256 to zigSources in flake.nix."
          else
            pkgs.stdenvNoCC.mkDerivation {
              pname = "zig";
              version = "0.16.0";
              src = pkgs.fetchurl {
                inherit (zigSrc) url sha256;
              };
              dontConfigure = true;
              dontBuild = true;
              dontFixup = true;
              installPhase = ''
                runHook preInstall
                mkdir -p $out/{bin,doc,lib}
                cp -r ./lib/* $out/lib/
                cp -r ./docs/* $out/doc/ 2>/dev/null || true
                install -Dm755 ./zig $out/bin/zig
                runHook postInstall
              '';
              meta = with pkgs.lib; {
                description = "General-purpose programming language and toolchain (0.16.0, pinned)";
                homepage = "https://ziglang.org/";
                license = licenses.mit;
                platforms = builtins.attrNames zigSources;
              };
            };

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [ websockets ]);

        commonDeps = [
          zig_0_16_0
          pkgs.jq
          pkgs.curl
          pkgs.gawk
          pythonEnv
          pkgs.git
          pkgs.bashInteractive
          pkgs.coreutils
          pkgs.gnused
        ];
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = commonDeps;
          shellHook = ''
            echo "tokenizers-zig devShell — Zig $(zig version), jq $(jq --version), python $(python3 --version 2>&1)"
            echo "Run: zig build test"
          '';
        };

        packages.zig = zig_0_16_0;

        # nix run .#test  → zig build test
        apps.test = {
          type = "app";
          program = toString (pkgs.writeShellScript "tokenizers-zig-test" ''
            set -euo pipefail
            export PATH=${pkgs.lib.makeBinPath commonDeps}:$PATH
            cd "$(${pkgs.git}/bin/git rev-parse --show-toplevel)"
            ${zig_0_16_0}/bin/zig build test --summary all
          '');
        };
      });
}
