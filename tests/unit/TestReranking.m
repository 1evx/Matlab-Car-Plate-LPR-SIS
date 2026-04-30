classdef TestReranking < matlab.unittest.TestCase
    methods (Test)
        function regexValidCandidateOutranksEmptyBackground(testCase)
            cfg = defaultConfig();
            testCase.assumeTrue(exist("ocr", "file") == 2 || exist("ocr", "builtin") == 5, ...
                "MATLAB OCR function is unavailable.");

            [sceneImage, meta] = createSyntheticPlateImage("BPK1234");
            detectorCandidates = [ ...
                struct("bbox", [20 30 140 55], "score", 0.95, "branchName", "edge_full", "profileName", "single_line", "scale", 1.0), ...
                struct("bbox", meta.plateBox, "score", 0.60, "branchName", "text_priority", "profileName", "single_line", "scale", 1.0)];

            [selectedIndex, evaluatedCandidates] = rerankPlateCandidates(sceneImage, detectorCandidates, cfg);

            testCase.verifyEqual(selectedIndex, 1);
            testCase.verifyEqual(evaluatedCandidates(1).rawBBox, meta.plateBox);
            testCase.verifyGreaterThan(evaluatedCandidates(1).finalScore, evaluatedCandidates(2).finalScore);
            testCase.verifyTrue(isfield(evaluatedCandidates, "characterTextureScore"));
            testCase.verifyTrue(isfield(evaluatedCandidates, "plateEvidenceScore"));
            testCase.verifyTrue(isfield(evaluatedCandidates, "emptyPenalty"));
            testCase.verifyTrue(isfield(evaluatedCandidates, "lengthScore"));
            testCase.verifyTrue(isfield(evaluatedCandidates, "recognitionPath"));
            testCase.verifyEqual(string(evaluatedCandidates(1).recognitionPath), "matlab_ocr");
            testCase.verifyGreaterThan(evaluatedCandidates(2).emptyPenalty, 0);
            testCase.verifyTrue(isfield(evaluatedCandidates(1).scoreBreakdown, "plateEvidence"));
            testCase.verifyTrue(isfield(evaluatedCandidates(1).scoreBreakdown, "length"));
            testCase.verifyTrue(isfield(evaluatedCandidates(1).scoreBreakdown, "structure"));
            testCase.verifyTrue(isfield(evaluatedCandidates(1).scoreBreakdown, "framing"));
            testCase.verifyTrue(isfield(evaluatedCandidates, "structureScore"));
            testCase.verifyTrue(isfield(evaluatedCandidates, "framingScore"));
        end
    end
end
