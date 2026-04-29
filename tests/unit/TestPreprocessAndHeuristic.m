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
    end
end
