# micropolis-4bots #

This fork makes a few small modifications to MicropolisCore, making the engine available to python for stepping through the simulation with or without drawing the gtk window, and removing the handleDidTool() printout which would otherwise clutter the terminal during training. It is intended for use with the [micropolis gym environment](https://github.com/smearle/gym-micropolis). Note that only the MicropolisCore module oought to be installed to these ends, the rest of the repo, including old micropolis-activity, is included only for posterity, entirely unchanged.

## TODO: ##

* Might it be faster to access the simulation's map/tile-array directly for agent observation, rather than leaving agents to maintain an internal representation given the success/fail of their own actions? If not, then we should:
 - have toolDown() method cause bot to update its internal representations, when toolDown() has been caused by something other than the bot itself
* Make available any other potentially interesting simulation variables for bots
* Make pre-trained city-building bots available in-game, as player assistants/advisors

Huge thank you to Don Hopkins, and to all the ardent souls who made this brilliant code available for the manglin'.
Find the original repo's readthrough below.

# Open Source Micropolis, based on the original SimCity Classic from Maxis, by Will Wright. #

This is the source code for Micropolis (based on [SimCity](http://en.wikipedia.org/wiki/SimCity_(1989_video_game))), released under the GPL. Micropolis is based on the original SimCity from Electronic Arts / Maxis, and designed and written by Will Wright.

## [Description](../wiki/Description.md) ##
A description of the Micropolis project source code release.

## [News](../wiki/News.md) ##
The latest news about recent development.

## [DevelopmentPlan](../wiki/DevelopmentPlan.md) ##
The development plan, and a high level description of tasks that need to be done.

## [ThePlan](../wiki/ThePlan.md) ##
Older development plan for the TCL/Tk version of Micropolis and the C++/Python version too.

## [Assets](../wiki/Assets.md) ##
List of art and text assets, and work that needs to be done for Micropolis.

## Documentation ##

This is the old documentation of the HyperLook version of SimCity, converted to wiki text.
It needs to be brought up to date and illustrated.

  * [Introduction](../wiki/Introduction.md)
  * [Tutorial](../wiki/Tutorial.md)
  * [User Reference](../wiki/UserReference.md)
  * [Inside The Simulator](../wiki/InsideTheSimulator.md)
  * [History Of Cities And City Planning](../wiki/History.md)
  * [Bibliography](../wiki/Bibliography.md)
  * [Credits](../wiki/Credits.md)

## [License](../wiki/License.md) ##
The Micropolis GPL license.

## Tools ##
[![](http://wingware.com/images/coded-with-logo-129x66.png)](http://wingware.com/)
