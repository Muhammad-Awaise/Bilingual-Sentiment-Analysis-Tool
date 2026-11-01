# Linguistic Error Analysis

These examples probe negation, contrast, code-switching, intensifiers,
and informal Roman Urdu spelling. They test application behavior and
provide a framework for qualitative error analysis.

**They are not proof that the trained model classifies every sentence correctly.**

- Examples checked: 15
- Matches expected linguistic label: 15/15

| Category | Text | Expected | Predicted | Match | Notes |
|----------|------|----------|-----------|-------|-------|
| negation_control | This product is good | positive | positive | True | Baseline positive without negation |
| negation_english | This product is not good | negative | negative | True | English negation flips polarity |
| negation_control_urdu | Product acha hai | positive | positive | True | Roman Urdu positive baseline |
| negation_roman_urdu | Product acha nahi hai | negative | negative | True | Roman Urdu negation with nahi |
| negation_spelling | Product acha nahin hai | negative | negative | True | Alternate spelling nahin |
| contrast_lekin | Product acha hai lekin quality kharab hai | negative | negative | True | Contrast clause with lekin dominates overall judgment |
| contrast_code_switch | Quality achi hai but delivery late thi | negative | negative | True | Code-switched contrast with but |
| intensifier_positive | Bohat acha product hai | positive | positive | True | Intensifier bohat strengthens positive |
| intensifier_negative | Bohat bura experience tha | negative | negative | True | Intensifier bohat strengthens negative |
| informal_spelling_pos | Qlty zabrdast hai price bhi ok | positive | positive | True | Compressed informal Roman Urdu spelling |
| informal_spelling_neg | Qlty kharab hai price bhi ziada | negative | negative | True | Informal spelling with negative cues |
| mixed_ambiguous | I like the design but I am not happy with the battery | negative | negative | True | Mixed sentiment; overall dissatisfaction on key feature |
| negation_of_complaint | No complaints everything works well | positive | positive | True | Negation of a negative noun yields positive meaning |
| strong_neg_code_mixed | Kabhi nahi lunga dobara worst product | negative | negative | True | Strong negative with code-mixing |
| code_switch_positive | Seller cooperative tha and item exact same as photos | positive | positive | True | Positive code-switched review |

## How to read mismatches

- Small training data and spelling variation often cause failures on negation scope.
- Contrastive 'but' / 'lekin' clauses may leave residual positive lexical cues.
- Annotation disagreement is possible even for humans on mixed sentiment.
