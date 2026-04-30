classdef TestDetection < matlab.unittest.TestCase
    methods (Test)
        function detectionReturnsIntermediateMasks(testCase)
            cfg = defaultConfig();
            [sceneImage, ~] = createSyntheticPlateImage("BPK1234");
            grayImage = im2gray(sceneImage);

            [bbox, metadata] = detectPlateRegion(grayImage, cfg);

            testCase.verifyNotEmpty(bbox);
            testCase.verifyNotEmpty(metadata.edgeMask);
            testCase.verifyNotEmpty(metadata.closedMask);
            testCase.verifyNotEmpty(metadata.openedMask);
            testCase.verifyNotEmpty(metadata.dilatedMask);
            testCase.verifyNotEmpty(metadata.plateMask);
            testCase.verifyNotEmpty(metadata.priorityEdgeMask);
            testCase.verifyNotEmpty(metadata.darkPlateMask);
            testCase.verifyNotEmpty(metadata.textClusterMask);
            testCase.verifyNotEmpty(metadata.candidates);
            testCase.verifyNotEmpty(metadata.topCandidates);
            testCase.verifyTrue(isfield(metadata.candidates, "profileName"));
            testCase.verifyTrue(isfield(metadata.candidates, "branchName"));
            testCase.verifyTrue(isfield(metadata.candidates, "scoreBreakdown"));
            testCase.verifyTrue(isfield(metadata.candidates, "characterTextureScore"));
            testCase.verifyTrue(isfield(metadata.candidates, "plateContrastScore"));
            testCase.verifyTrue(isfield(metadata.candidates, "componentAlignmentScore"));
            testCase.verifyTrue(isfield(metadata.candidates(1).scoreBreakdown, "textBandScore"));
            testCase.verifyTrue(isfield(metadata.candidates(1).scoreBreakdown, "edgeClipPenalty"));
            testCase.verifyFalse(isfield(metadata.candidates, "vehiclePositionScore"));
        end
    end
end
