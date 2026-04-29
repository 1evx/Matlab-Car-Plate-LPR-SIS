classdef TestPreprocessAndHeuristic < matlab.unittest.TestCase
    methods (Test)
        function preprocessImageReturnsSingleChannelOutput(testCase)
            cfg = defaultConfig();
            [sceneImage, ~] = createSyntheticPlateImage("BPK1234");

            [preprocessedImage, metadata] = preprocessImage(sceneImage, cfg);

            testCase.verifyEqual(ndims(preprocessedImage), 2);
            testCase.verifySize(preprocessedImage, size(metadata.grayImage));
            testCase.verifyNotEmpty(metadata.preprocessedImage);
        end

        function heuristicReturnsExpectedStructFields(testCase)
            cfg = defaultConfig();
            [sceneImage, meta] = createSyntheticPlateImage("BPK1234");

            estimate = estimateVehicleTypeHeuristic(sceneImage, meta.plateBox, cfg);

            testCase.verifyTrue(isfield(estimate, "label"));
            testCase.verifyTrue(isfield(estimate, "confidence"));
            testCase.verifyTrue(isfield(estimate, "reason"));
            testCase.verifyTrue(isfield(estimate, "metrics"));
        end
    end
end
