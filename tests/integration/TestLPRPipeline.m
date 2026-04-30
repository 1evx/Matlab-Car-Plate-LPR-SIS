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

        function pipelineHandlesDoubleLineSyntheticScene(testCase)
            cfg = defaultConfig();
            testCase.assumeTrue(exist("ocr", "file") == 2 || exist("ocr", "builtin") == 5, ...
                "MATLAB OCR function is unavailable.");

            [sceneImage, ~] = createSyntheticTwoRowPlateImage("BPK", "1234");
            result = runLPRPipeline(sceneImage, cfg);

            testCase.verifyNotEmpty(result.plateBBox);
            testCase.verifyTrue(isfield(result.debug, "rowCompositePlate"));
            testCase.verifyNotEmpty(result.debug.rowCompositePlate);
            testCase.verifyEqual(string(result.debug.rectifiedLayoutHint), "two_row");
            testCase.verifyNotEqual(result.status, "plate_not_found");
            testCase.verifyGreaterThan(strlength(result.recognizedText), 0);
        end

        function pipelineImprovesTinySyntheticPlateCrop(testCase)
            cfg = defaultConfig();
            testCase.assumeTrue(exist("ocr", "file") == 2 || exist("ocr", "builtin") == 5, ...
                "MATLAB OCR function is unavailable.");

            [sceneImage, ~] = createSyntheticTinyPlateImage("WVU8899");
            result = runLPRPipeline(sceneImage, cfg);

            testCase.verifyNotEmpty(result.plateBBox);
            testCase.verifyTrue(isfield(result.debug, "textOnlyPlate"));
            testCase.verifyNotEmpty(result.debug.textOnlyPlate);
            testCase.verifyGreaterThanOrEqual(size(result.debug.textOnlyPlate, 1), ...
                cfg.rectification.minTextHeightPixels);
            testCase.verifyNotEqual(result.status, "plate_not_found");
        end

        function pipelineAvoidsWeakHallucinatedSuffixOnRealImage(testCase)
            cfg = defaultConfig();
            testCase.assumeTrue(exist("ocr", "file") == 2 || exist("ocr", "builtin") == 5, ...
                "MATLAB OCR function is unavailable.");

            testRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            imageFiles = dir(fullfile(testRoot, "data", "raw", "cars", "*JRC5492*"));
            testCase.assumeNotEmpty(imageFiles, "Real-image regression file JRC5492 is unavailable.");

            imagePath = fullfile(imageFiles(1).folder, imageFiles(1).name);
            result = runLPRPipeline(imagePath, cfg);

            testCase.verifyEqual(string(result.recognizedText), "JRC5492");
        end
    end
end
