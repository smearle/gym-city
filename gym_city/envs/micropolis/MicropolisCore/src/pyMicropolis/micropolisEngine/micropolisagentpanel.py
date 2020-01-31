# micropolisevaluationpanel.py
#
# Micropolis, Unix Version.  This game was released for the Unix platform
# in or about 1990 and has been modified for inclusion in the One Laptop
# Per Child program.  Copyright (C) 1989 - 2007 Electronic Arts Inc.  If
# you need assistance with this program, you may contact:
#   http://wiki.laptop.org/go/Micropolis  or email  micropolis@laptop.org.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.  You should have received a
# copy of the GNU General Public License along with this program.  If
# not, see <http://www.gnu.org/licenses/>.
#
#             ADDITIONAL TERMS per GNU GPL Section 7
#
# No trademark or publicity rights are granted.  This license does NOT
# give you any right, title or interest in the trademark SimCity or any
# other Electronic Arts trademark.  You may not distribute any
# modification of this program using the trademark SimCity or claim any
# affliation or association with Electronic Arts Inc. or its employees.
#
# Any propagation or conveyance of this program must include this
# copyright notice and these terms.
#
# If you convey this program (or any modifications of it) and assume
# contractual liability for the program to recipients of it, you agree
# to indemnify Electronic Arts for any liability that those contractual
# assumptions impose on Electronic Arts.
#
# You may not misrepresent the origins of this program; modified
# versions of the program must be marked as such and not identified as
# the original program.
#
# This disclaimer supplements the one included in the General Public
# License.  TO THE FULLEST EXTENT PERMISSIBLE UNDER APPLICABLE LAW, THIS
# PROGRAM IS PROVIDED TO YOU "AS IS," WITH ALL FAULTS, WITHOUT WARRANTY
# OF ANY KIND, AND YOUR USE IS AT YOUR SOLE RISK.  THE ENTIRE RISK OF
# SATISFACTORY QUALITY AND PERFORMANCE RESIDES WITH YOU.  ELECTRONIC ARTS
# DISCLAIMS ANY AND ALL EXPRESS, IMPLIED OR STATUTORY WARRANTIES,
# INCLUDING IMPLIED WARRANTIES OF MERCHANTABILITY, SATISFACTORY QUALITY,
# FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT OF THIRD PARTY
# RIGHTS, AND WARRANTIES (IF ANY) ARISING FROM A COURSE OF DEALING,
# USAGE, OR TRADE PRACTICE.  ELECTRONIC ARTS DOES NOT WARRANT AGAINST
# INTERFERENCE WITH YOUR ENJOYMENT OF THE PROGRAM; THAT THE PROGRAM WILL
# MEET YOUR REQUIREMENTS; THAT OPERATION OF THE PROGRAM WILL BE
# UNINTERRUPTED OR ERROR-FREE, OR THAT THE PROGRAM WILL BE COMPATIBLE
# WITH THIRD PARTY SOFTWARE OR THAT ANY ERRORS IN THE PROGRAM WILL BE
# CORRECTED.  NO ORAL OR WRITTEN ADVICE PROVIDED BY ELECTRONIC ARTS OR
# ANY AUTHORIZED REPRESENTATIVE SHALL CREATE A WARRANTY.  SOME
# JURISDICTIONS DO NOT ALLOW THE EXCLUSION OF OR LIMITATIONS ON IMPLIED
# WARRANTIES OR THE LIMITATIONS ON THE APPLICABLE STATUTORY RIGHTS OF A
# CONSUMER, SO SOME OR ALL OF THE ABOVE EXCLUSIONS AND LIMITATIONS MAY
# NOT APPLY TO YOU.


########################################################################
# Micropolis Evaluation Panel
# Don Hopkins


########################################################################
# Import stuff


from gi.repository import Gtk as gtk
import cairo
from gi.repository import Pango as pango
from . import micropolisengine
from . import micropolisview


########################################################################
# MicropolisEvaluationPanel


class MicropolisAgentPanel(gtk.Frame):



    def __init__(
        self,
        engine=None,
        **args):

        gtk.Frame.__init__(
            self,
            **args)

        self.engine = engine

        # Views

        hbox1 = gtk.HBox(False, 0)
        self.hbox1 = hbox1
        self.add(hbox1)

        vbox1 = gtk.VBox(False, 0)
        self.vbox1 = vbox1
        hbox1.pack_start(vbox1, False, False, 0)
        vbox2 = gtk.VBox(False, 0)
        self.vbox2 = vbox2
        hbox1.pack_start(vbox2, False, False, 0)
        vbox3 = gtk.VBox(False, 0)
        self.vbox3 = vbox3
        hbox1.pack_start(vbox3, False, False, 0)

        buttonMonster = gtk.Button("Reset")
        self.buttonMonster = buttonMonster
        buttonMonster.connect('clicked', lambda item: self.reset_game())
        vbox1.pack_start(buttonMonster, False, False, 0)

        checkButtonAutoReset = gtk.CheckButton("Auto Reset")
        self.checkButtonAutoReset = checkButtonAutoReset
        checkButtonAutoReset.connect('clicked', lambda item: self.enable_auto_reset())
        vbox1.pack_start(checkButtonAutoReset, False, False, 0)
        self.checkButtonAutoReset.set_active(True)

        self.checkButtonStaticBuilds = gtk.CheckButton("Static Builds")
        self.checkButtonStaticBuilds.connect('toggled', lambda item: self.set_static())
        self.vbox1.pack_start(self.checkButtonStaticBuilds, False, False, 0)

        #TODO: not this
        pop_threshold = (0, 1000)
        traffic_range = (0, 2000)
        num_plants_range = (0, 200)
        mayor_rating_range = (0, 100)

        scaleRes = gtk.HScale()
        scaleResMetric = gtk.HScale()
        self.scaleRes = scaleRes
        self.scaleResMetric = scaleResMetric
        scaleRes.set_digits(10)
        scaleResMetric.set_digits(10)
        scaleRes.set_range(*pop_threshold)
        scaleResMetric.set_range(*pop_threshold)
        scaleRes.connect('value_changed', self.scaleResChanged)
        labelRes = gtk.Label('Residential:')
        vbox2.pack_start(labelRes, False, False, 0)
        vbox2.pack_start(scaleRes, False, False, 0)
        vbox2.pack_start(scaleResMetric, False, False, 0)

        scaleCom = gtk.HScale()
        scaleComMetric = gtk.HScale()
        self.scaleCom = scaleCom
        self.scaleComMetric = scaleComMetric
        scaleCom.set_digits(10)
        scaleCom.set_range(*pop_threshold)
        scaleCom.connect('value_changed', self.scaleComChanged)
        scaleComMetric.set_digits(10)
        scaleComMetric.set_range(*pop_threshold)
        labelCom = gtk.Label('Commercial:')
        vbox2.pack_start(labelCom, False, False, 0)
        vbox2.pack_start(scaleCom, False, False, 0)
        vbox2.pack_start(scaleComMetric, False, False, 0)

        scaleInd = gtk.HScale()
        self.scaleInd = scaleInd
        scaleInd.set_digits(10)
        scaleInd.set_range(*pop_threshold)
        scaleIndMetric = gtk.HScale()
        self.scaleIndMetric = scaleIndMetric
        scaleIndMetric.set_digits(10)
        scaleIndMetric.set_range(*pop_threshold)
        scaleInd.connect('value_changed', self.scaleIndChanged)
        labelInd = gtk.Label('Industrial:')
        vbox2.pack_start(labelInd, False, False, 0)
        vbox2.pack_start(scaleInd, False, False, 0)
        vbox2.pack_start(scaleIndMetric, False, False, 0)

        scaleTraffic = gtk.HScale()
        self.scaleTraffic = scaleTraffic
        scaleTraffic.set_digits(10)
        scaleTraffic.set_range(0, 2000)
        scaleTraffic.connect('value_changed', self.scaleTrafficChanged)
        scaleTrafficMetric = gtk.HScale()
        self.scaleTrafficMetric = scaleTrafficMetric
        scaleTrafficMetric.set_digits(10)
        scaleTrafficMetric.set_range(0, 2000)
        labelTraffic = gtk.Label('Traffic:')
        vbox3.pack_start(labelTraffic, False, False, 0)
        vbox3.pack_start(scaleTraffic, False, False, 0)
        vbox3.pack_start(scaleTrafficMetric, False, False, 0)

        scalePlants = gtk.HScale()
        self.scalePlants = scalePlants
        scalePlants.set_digits(10)
        scalePlants.set_range(*num_plants_range)
        scalePlantsMetric = gtk.HScale()
        self.scalePlantsMetric = scalePlantsMetric
        scalePlantsMetric.set_digits(10)
        scalePlantsMetric.set_range(*num_plants_range)
        scalePlants.connect('value_changed', self.scalePlantsChanged)
        labelPlants = gtk.Label('Plants:')
        vbox3.pack_start(labelPlants, False, False, 0)
        vbox3.pack_start(scalePlants, False, False, 0)
        vbox3.pack_start(scalePlantsMetric, False, False, 0)

        scaleRating = gtk.HScale()
        self.scaleRating = scaleRating
        scaleRating.set_digits(10)
        scaleRating.set_range(*mayor_rating_range)
        scaleRatingMetric = gtk.HScale()
        self.scaleRatingMetric = scaleRatingMetric
        scaleRatingMetric.set_digits(10)
        scaleRatingMetric.set_range(*mayor_rating_range)
        scaleRating.connect('value_changed', self.scaleRatingChanged)
        labelRating = gtk.Label('Rating:')
        vbox3.pack_start(labelRating, False, False, 0)
        vbox3.pack_start(scaleRating, False, False, 0)
        vbox3.pack_start(scaleRatingMetric, False, False, 0)



    def scaleResChanged(self, scale):
        self.engine.env.set_res_weight(scale.get_value())

    def scaleComChanged(self, scale):
        self.engine.env.set_com_weight(scale.get_value())

    def scaleIndChanged(self, scale):
        self.engine.env.set_ind_weight(scale.get_value())

    def scaleTrafficChanged(self, scale):
        self.engine.env.set_traffic_weight(scale.get_value())

    def scalePlantsChanged(self, scale):
        self.engine.env.set_plants_weight(scale.get_value())

    def scaleRatingChanged(self, scale):
        self.engine.env.set_rating_weight(scale.get_value())

    def displayTrgs(self, trgs):
        self.scaleRes.set_value(trgs['res_pop'])
        self.scaleCom.set_value(trgs['com_pop'])
        self.scaleInd.set_value(trgs['ind_pop'])
        self.scaleTraffic.set_value(trgs['traffic'])
        self.scalePlants.set_value(trgs['num_plants'])
        self.scaleRating.set_value(trgs['mayor_rating'])

    def displayMetrics(self, metrics):
        self.scaleResMetric.set_value(metrics['res_pop'])
        self.scaleComMetric.set_value(metrics['com_pop'])
        self.scaleIndMetric.set_value(metrics['ind_pop'])
        self.scaleTrafficMetric.set_value(metrics['traffic'])
        self.scalePlantsMetric.set_value(metrics['num_plants'])
        self.scaleRatingMetric.set_value(metrics['mayor_rating'])

    def setMetricRanges(self, metric_ranges):
        self.scaleResMetric.set_range(*metric_ranges['res_pop'])
        self.scaleComMetric.set_range(*metric_ranges['com_pop'])
        self.scaleIndMetric.set_range(*metric_ranges['ind_pop'])
        self.scaleTrafficMetric.set_range(*metric_ranges['traffic'])
        self.scalePlantsMetric.set_range(*metric_ranges['num_plants'])
        self.scaleRes.set_range(*metric_ranges['res_pop'])
        self.scaleCom.set_range(*metric_ranges['com_pop'])
        self.scaleInd.set_range(*metric_ranges['ind_pop'])
        self.scaleTraffic.set_range(*metric_ranges['traffic'])
        self.scalePlants.set_range(*metric_ranges['num_plants'])
        self.scaleRating.set_range(*metric_ranges['mayor_rating'])
        self.scaleRating.set_range(*metric_ranges['mayor_rating'])

    def reset_game(self):
        self.engine.env.reset()

    def set_static(self):
        self.engine.env.static_player_builds = self.checkButtonStaticBuilds.get_active()

    def enable_auto_reset(self):
        self.engine.env.auto_reset = self.checkButtonAutoReset.get_active()


########################################################################
