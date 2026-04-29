function startup()
    %STARTUP Configure paths and environment for the LPR project.
    
    rootDir = fileparts(mfilename("fullpath"));
    projectPaths = {
        fullfile(rootDir, "src")
        fullfile(rootDir, "tests")
        fullfile(rootDir, "experiments")
        };
    
    for i = 1:numel(projectPaths)
        if isfolder(projectPaths{i})
            addpath(genpath(projectPaths{i}));
        end
    end
    
    setappdata(0, "CarPlateRecogniseSystemRoot", rootDir);
    cfg = defaultConfig(rootDir);
    setappdata(0, "CarPlateRecogniseSystemConfig", cfg);
    
    fprintf("[startup] Project root: %s\n", rootDir);
    fprintf("[startup] Ready. Launch the app with: app = LPRStateApp; or run: main\n");
end
