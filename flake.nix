{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (system: {
        default = pkgs.${system}.mkShellNoCC {
          packages = with pkgs.${system}; [
            (python312.withPackages (ppkgs: [
              ppkgs.manim
              ppkgs.jupyter
              ppkgs.numpy
              ppkgs.scipy
            ]))
            ffmpeg-full
            wgo
          ];

          LD_LIBRARY_PATH = "${pkgs.${system}.libGL}/lib";
        };
      });
    };
}
