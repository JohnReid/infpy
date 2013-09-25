#
# Copyright John Reid 2007, 2010
#


import unittest, logging
from infpy.decision import EnumerativeAttribute, OrdinalAttribute, new_rule_for_attribute

class RuleTest( unittest.TestCase ):
    """Test case for decision tree rule generation"""

    def test_rule_generation(self):
        "Test decision tree rule generation."
        def a(data): return 1
        def b(data): return 1
        class TestData(object): pass

        enum_attr = EnumerativeAttribute('Enumerative attribute', a, 3)
        rule, _num_values = new_rule_for_attribute(enum_attr)
        logging.debug(rule(TestData()))

        ord_attr = OrdinalAttribute('Ordinal attribute 1', b, 3)
        rule, _num_values = new_rule_for_attribute(ord_attr)
        logging.debug(rule(TestData()))

        ord_attr = OrdinalAttribute('Ordinal attribute 2', b, 3)
        rule, _num_values = new_rule_for_attribute(ord_attr)
        logging.debug(rule(TestData()))

        ord_attr = OrdinalAttribute('Ordinal attribute 3', b, 3)
        rule, _num_values = new_rule_for_attribute(ord_attr)
        logging.debug(rule(TestData()))




if __name__ == "__main__":
    unittest.main()
