import unittest
import numpy as np
from tsr import tsr
from fox_toolbox.utils import xml_parser
from tsr import csv_parser


class TestNotebook(unittest.TestCase):

    def setUp(self):
        cms_xml = xml_parser.get_files('irsmform xml', folder = r'notebooks\linear TSR logs')
        cms_csv = xml_parser.get_files('csv', folder = r'notebooks\linear TSR logs')
        
        self.cms_replic_basket = csv_parser.parse_csv(cms_csv)
        self.cal_basket = list(xml_parser.get_calib_basket(cms_xml))
           

    def test_tsr_strikes(self):
        for (caplet, floorlet), swo in zip(self.cms_replic_basket, self.cal_basket):
            self.assertAlmostEqual(tsr.strike_max(swo.vol.value, caplet.fixing_date, caplet.fwd, caplet.n), caplet.strike_max,\
                                   places=11, msg='tsr caplet strike is not reconciled')
            self.assertAlmostEqual(tsr.strike_min(swo.vol.value, floorlet.fixing_date, floorlet.fwd, floorlet.n), floorlet.strike_min,\
                                   places=11, msg='tsr floorlet strike is not reconciled')



if __name__ == '__main__':
    unittest.main()
