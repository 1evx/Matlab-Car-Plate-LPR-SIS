classdef TestRecognition < matlab.unittest.TestCase
    methods (Test)
        function matlabRecognizerReadsPlateCrop(testCase)
            cfg = defaultConfig();
            testCase.assumeTrue(exist("ocr", "file") == 2 || exist("ocr", "builtin") == 5, ...
                "MATLAB OCR function is unavailable.");

            [sceneImage, meta] = createSyntheticPlateImage("BPK1234");
            plateCrop = imcrop(sceneImage, meta.plateBox);

            [recognizedText, metadata] = recognizeCharacters(plateCrop, cfg);

            testCase.verifyEqual(string(metadata.method), "matlab_ocr");
            testCase.verifyTrue(isfield(metadata, "ocrInputPlate"));
            testCase.verifyTrue(isfield(metadata, "matlabOcr"));
            testCase.verifyTrue(metadata.matlabOcr.success);
            testCase.verifyEqual(string(recognizedText), "BPK1234");
            testCase.verifyGreaterThanOrEqual(mean(metadata.confidences), 0);
        end

        function matlabRecognizerReturnsEmptyTextForEmptyPlate(testCase)
            cfg = defaultConfig();
            plateCrop = uint8([]);

            [recognizedText, metadata] = recognizeCharacters(plateCrop, cfg);

            testCase.verifyEqual(string(recognizedText), "");
            testCase.verifyEqual(string(metadata.method), "matlab_ocr");
            testCase.verifyTrue(isfield(metadata, "matlabOcr"));
            testCase.verifyFalse(metadata.matlabOcr.success);
        end
    end
end
