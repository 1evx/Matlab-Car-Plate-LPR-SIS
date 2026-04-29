classdef TestReranking < matlab.unittest.TestCase
    methods (Test)
        function regexValidCandidateOutranksEmptyBackground(testCase)
            cfg = defaultConfig();
            cfg.classification.syntheticFonts = {'Consolas'};
            cfg.classification.syntheticRotationDegrees = 0;
            cfg.classification.trainedModelPath = fullfile(tempdir, "baselineCharacterModel_rerank.mat");

            [sceneImage, meta] = createSyntheticPlateImage("BPK1234");
            detectorCandidates = [ ...
                struct("bbox", [20 30 140 55], "score", 0.95, "branchName", "edge_full", "profileName", "single_line", "scale", 1.0), ...
                struct("bbox", meta.plateBox, "score", 0.60, "branchName", "text_priority", "profileName", "single_line", "scale", 1.0)];

            [selectedIndex, evaluatedCandidates] = rerankPlateCandidates(sceneImage, detectorCandidates, cfg);

            testCase.verifyEqual(selectedIndex, 1);
            testCase.verifyEqual(evaluatedCandidates(1).rawBBox, meta.plateBox);
            testCase.verifyGreaterThan(evaluatedCandidates(1).finalScore, evaluatedCandidates(2).finalScore);
            testCase.verifyGreaterThanOrEqual(evaluatedCandidates(1).characterCount, 2);
            testCase.verifyTrue(isfield(evaluatedCandidates, "characterTextureScore"));
            testCase.verifyTrue(isfield(evaluatedCandidates, "segmentationScore"));
            testCase.verifyTrue(isfield(evaluatedCandidates, "emptyPenalty"));
            testCase.verifyGreaterThan(evaluatedCandidates(2).emptyPenalty, 0);
            testCase.verifyTrue(isfield(evaluatedCandidates(1).scoreBreakdown, "plateEvidence"));
        end
    end
end
