classdef TestLPRPipeline < matlab.unittest.TestCase
    methods (Test)
        function pipelineRunsEndToEndOnSyntheticScene(testCase)
            cfg = defaultConfig();
            cfg.classification.syntheticFonts = {'Consolas'};
            cfg.classification.syntheticRotationDegrees = 0;

            [sceneImage, ~] = createSyntheticPlateImage("BPK1234");
            result = runLPRPipeline(sceneImage, cfg);

            testCase.verifyNotEmpty(result.plateBBox);
            testCase.verifyEqual(result.status, "ok");
            testCase.verifyGreaterThanOrEqual(numel(result.characterImages), 2);
            testCase.verifyNotEmpty(result.stateInfo.name);
            testCase.verifyTrue(isfield(result, "estimatedVehicleType"));
            testCase.verifyTrue(isfield(result.debug, "evaluatedCandidates"));
            testCase.verifyNotEmpty(result.debug.evaluatedCandidates);
            testCase.verifyTrue(isfield(result.debug, "selectedCandidateIndex"));
        end

        function guiCanBeConstructedWithoutErrors(testCase)
            app = LPRStateApp(defaultConfig(), struct("Visible", "off"));
            cleaner = onCleanup(@() delete(app)); %#ok<NASGU>

            testCase.verifyClass(app, "LPRStateApp");
        end
    end
end
