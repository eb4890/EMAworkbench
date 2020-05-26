'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''

from __future__ import (print_function, absolute_import, unicode_literals, 
                        division)

import unittest

import unittest.mock as mock


from ema_workbench.em_framework.model import MultiModel, SplitModel
from ema_workbench.em_framework.parameters import (RealParameter, Policy, 
                                                   Scenario, Category,
                                                   CategoricalParameter)
from ema_workbench.util import EMAError

class TestSplitAndMultiModel(unittest.TestCase):

    def test_init(self):
        multi_model_name = 'multimodelname'
        
        multi_model = MultiModel(multi_model_name)
        
        self.assertEqual(multi_model.name, multi_model_name)
        self.assertRaises(EMAError, MultiModel, '', 'model name')
        

    def test_model_init(self):
        multi_model_name = 'multimodelname'
        split_model_name_one = 'splitmodelname1'
        split_model_name_two = 'splitmodelname2'
        def update_func(a=1):
            return a
        def setup_func(b=1):
            return {'a': b}
        def report_func(a=1):
            return {'d': a}
        def variant_setup_func(a=1):
            return {'a': a}
        def variant_report_func(b=1):
            return {'e': a}
        split_model = SplitModel(split_model_name_one,
                           update=update_func,
                           setup=setup_func,
                           report=report_func,
                           variant_setup=variant_setup_func,
                           variant_report=variant_report_func,
                           iterations = 1)


        self.assertEqual(split_model.update, update_func)
        self.assertEqual(split_model.setup, setup_func)
        self.assertEqual(split_model.report, report_func)
        self.assertEqual(split_model.variant_report, variant_report_func)
        self.assertEqual(split_model.variant_setup, variant_setup_func)
        self.assertEqual(split_model.iterations, 1)

        multi_model = MultiModel(multi_model_name)

        with self.assertRaises(AttributeError):
            split_model.unknown


    def test_run_model(self):

        multi_model_name = 'multimodelname'
        split_model_name_one = 'splitmodelname1'
        split_model_name_two = 'splitmodelname2'
        func = lambda a: {'a': 'a'}
        update_func = mock.Mock(side_effect = func)
        setup_func = mock.Mock(return_value={'a': 'a'})
        report_func = mock.Mock(side_effect = func)
        variant_setup_func = mock.Mock(side_effect = func)
        variant_report_func = mock.Mock(side_effect = func)

        def wrap_update_moc(a):
           return update_func(a)

        def wrap_report_moc(a):
           return report_func(a)

        def wrap_variant_setup_moc(a):
           return variant_setup_func(a)

        def wrap_variant_report_moc(a):
           return variant_report_func(a)
        
        split_model = SplitModel(split_model_name_one,
                           update=wrap_update_moc,
                           setup=setup_func,
                           report=wrap_report_moc,
                           variant_setup=wrap_variant_setup_moc,
                           variant_report=wrap_variant_report_moc,
                           iterations = 1,
                            num_variants=1)



        split_model.run_experiment({})
        update_func.assert_called_once_with('a')
        variant_report_func.assert_called_once_with('a')
        report_func.assert_called_once_with('a')
        setup_func.assert_called_once()
        variant_setup_func.assert_called_once_with('a')
        # test complete translation of scenario


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
