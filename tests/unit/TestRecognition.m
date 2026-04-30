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

        function recognizerEvaluatesRowCompositeVariantsForTwoRowPlate(testCase)
            cfg = defaultConfig();
            [sceneImage, meta] = createSyntheticTwoRowPlateImage("BPK", "1234");
            [rectifiedPlate, rectifyMeta] = rectifyPlate(sceneImage, meta.plateBox, cfg);

            [~, metadata] = recognizeCharacters(rectifiedPlate, cfg, struct( ...
                "profileName", "two_row", ...
                "layoutHint", rectifyMeta.layoutHint, ...
                "textOnlyPlate", rectifyMeta.textOnlyPlate, ...
                "rowCompositePlate", rectifyMeta.rowCompositePlate));

            attemptedNames = strings(numel(metadata.attemptedResults), 1);
            for i = 1:numel(metadata.attemptedResults)
                attemptedNames(i) = string(metadata.attemptedResults{i}.name);
            end
            testCase.verifyTrue(any(attemptedNames == "rowcomposite_line"));
            testCase.verifyTrue(any(attemptedNames == "textonly_block"));
        end

        function matlabRecognizerKeepsValidSuffixLetter(testCase)
            cfg = defaultConfig();
            testCase.assumeTrue(exist("ocr", "file") == 2 || exist("ocr", "builtin") == 5, ...
                "MATLAB OCR function is unavailable.");

            [sceneImage, meta] = createSyntheticPlateImage("SJ230R");
            plateCrop = imcrop(sceneImage, meta.plateBox);

            [recognizedText, metadata] = recognizeCharacters(plateCrop, cfg);

            testCase.verifyTrue(startsWith(string(recognizedText), "SJ"));
            testCase.verifyTrue(endsWith(string(recognizedText), "R"));
            testCase.verifyEqual(string(metadata.parsedPattern), "L2-D3-S1");
        end

        function recognizerAddsLeftTrimmedTextVariants(testCase)
            cfg = defaultConfig();
            [sceneImage, meta] = createSyntheticPlateImage("WC5763R");
            [rectifiedPlate, rectifyMeta] = rectifyPlate(sceneImage, meta.plateBox, cfg);

            [~, metadata] = recognizeCharacters(rectifiedPlate, cfg, struct( ...
                "layoutHint", "single_line", ...
                "textOnlyPlate", rectifyMeta.textOnlyPlate));

            attemptedNames = strings(numel(metadata.attemptedResults), 1);
            for i = 1:numel(metadata.attemptedResults)
                attemptedNames(i) = string(metadata.attemptedResults{i}.name);
            end
            testCase.verifyTrue(any(attemptedNames == "textonly_lefttrim_line"));
            testCase.verifyTrue(any(attemptedNames == "textonly_lefttrim_block"));
            testCase.verifyTrue(any(attemptedNames == "textonly_righttrim_line"));
            testCase.verifyTrue(any(attemptedNames == "textonly_righttrim_block"));
        end

        function prepareTesseractPlateImageHandlesTinyCrops(testCase)
            cfg = defaultConfig();
            tinyCrop = uint8(randi([0 255], 6, 7));

            preparedImage = prepareTesseractPlateImage(tinyCrop, cfg);

            testCase.verifyNotEmpty(preparedImage);
            testCase.verifyClass(preparedImage, "uint8");
            testCase.verifyGreaterThanOrEqual(size(preparedImage, 1), size(tinyCrop, 1));
            testCase.verifyGreaterThanOrEqual(size(preparedImage, 2), size(tinyCrop, 2));
        end
    end
end
