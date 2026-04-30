classdef TestConfigAndRules < matlab.unittest.TestCase
    methods (Test)
        function defaultConfigHasExpectedSections(testCase)
            cfg = defaultConfig();

            testCase.verifyTrue(isfield(cfg, "detection"));
            testCase.verifyTrue(isfield(cfg, "classification"));
            testCase.verifyTrue(isfield(cfg, "reranking"));
            testCase.verifyTrue(isfield(cfg.reranking.weights, "composition"));
            testCase.verifyTrue(isfield(cfg.debug, "showPreprocessingFigure"));
            testCase.verifyGreaterThan(numel(cfg.malaysiaRules), 5);
        end

        function stateMatcherRecognizesMalaysianPrefixes(testCase)
            rules = malaysiaPlateRules();

            selangor = identifyState("BPK1234", rules);
            kedah = identifyState("KAA9876", rules);
            military = identifyState("Z1234", rules);
            diplomatic = identifyState("12DC88", rules);

            testCase.verifyEqual(selangor.name, "Selangor");
            testCase.verifyEqual(kedah.name, "Kedah");
            testCase.verifyEqual(military.category, "Military");
            testCase.verifyEqual(diplomatic.category, "Diplomatic");
        end
    end
end
