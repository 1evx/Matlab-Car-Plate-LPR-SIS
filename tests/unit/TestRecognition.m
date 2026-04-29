classdef TestRecognition < matlab.unittest.TestCase
    methods (Test)
        function featureRecognizerReadsCleanRenderedGlyphs(testCase)
            cfg = defaultConfig();
            cfg.classification.syntheticFonts = {'Consolas'};
            cfg.classification.syntheticRotationDegrees = 0;
            cfg.classification.trainedModelPath = fullfile(tempdir, "baselineCharacterModel_test.mat");
            plateText = "B123";
            glyphs = cell(1, strlength(plateText));

            for i = 1:strlength(plateText)
                canvas = renderTextImage(extractBetween(plateText, i, i), [70 40], ...
                    "FontName", "Consolas", ...
                    "FontSize", 30);
                glyphs{i} = imbinarize(canvas, "adaptive", "ForegroundPolarity", "dark");
            end

            [recognizedText, metadata] = recognizeCharacters(glyphs, cfg, []);

            testCase.verifyEqual(string(metadata.method), "feature_classifier");
            testCase.verifyNumElements(metadata.confidences, strlength(plateText));
            testCase.verifyEqual(strlength(string(recognizedText)), strlength(plateText));
            testCase.verifyNumElements(metadata.topCandidates, strlength(plateText));
        end
    end
end
