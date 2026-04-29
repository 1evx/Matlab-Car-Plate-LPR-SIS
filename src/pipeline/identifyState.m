function stateInfo = identifyState(plateText, rules)
    %IDENTIFYSTATE Infer the Malaysian state or category from a recognized plate.

    if nargin < 2 || isempty(rules)
        rules = malaysiaPlateRules();
    end

    stateInfo = ruleBasedStateMatcher(plateText, rules);
end

