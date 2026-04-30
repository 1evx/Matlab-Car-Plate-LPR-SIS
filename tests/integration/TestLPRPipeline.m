classdef TestLPRPipeline < matlab.unittest.TestCase
    methods (Test)
        function pipelineRunsEndToEndOnSyntheticScene(testCase)
            cfg = defaultConfig();
            testCase.assumeTrue(exist("ocr", "file") == 2 || exist("ocr", "builtin") == 5, ...
                "MATLAB OCR function is unavailable.");

            [sceneImage, ~] = createSyntheticPlateImage("BPK1234");
            result = runLPRPipeline(sceneImage, cfg);

            testCase.verifyNotEmpty(result.plateBBox);
            testCase.verifyEqual(result.status, "ok");
            testCase.verifyFalse(isfield(result, "characterImages"));
            testCase.verifyFalse(isfield(result, "characterBBoxes"));
            testCase.verifyNotEmpty(result.stateInfo.name);
            testCase.verifyTrue(isfield(result, "recognitionPath"));
            testCase.verifyEqual(string(result.recognitionPath), "matlab_ocr");
            testCase.verifyTrue(isfield(result.debug, "evaluatedCandidates"));
            testCase.verifyNotEmpty(result.debug.evaluatedCandidates);
            testCase.verifyTrue(isfield(result.debug, "selectedCandidateIndex"));
            testCase.verifyTrue(isfield(result.debug, "recognition"));
            testCase.verifyTrue(isfield(result.debug, "ocrInputPlate"));
            testCase.verifyFalse(isfield(result.debug, "segmentedOverlay"));
        end

        function guiCanBeConstructedWithoutErrors(testCase)
            app = LPRStateApp(defaultConfig(), struct("Visible", "off"));
            cleaner = onCleanup(@() delete(app)); %#ok<NASGU>

            testCase.verifyClass(app, "LPRStateApp");
        end
    end
end
