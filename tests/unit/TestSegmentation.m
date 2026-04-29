classdef TestSegmentation < matlab.unittest.TestCase
    methods (Test)
        function segmentationFindsCharacterCandidates(testCase)
            cfg = defaultConfig();

            [sceneImage, meta] = createSyntheticPlateImage("BPK1234");
            plateCrop = imcrop(sceneImage, meta.plateBox);

            [characterImages, boxes, metadata] = segmentCharacters(plateCrop, cfg);

            testCase.verifyGreaterThanOrEqual(numel(characterImages), 2);
            testCase.verifyEqual(size(boxes, 2), 4);
            testCase.verifyNotEmpty(metadata.binaryMask);
        end
    end
end
