# Outsmart

Outsmart is a didactic puzzle game in which the player has to lure a robot into a trap, using the robot's own learning abilities against itself.

![Game screenshot](https://github.com/edouardklein/Outsmart/raw/master/img/screen.png "Game Screenshot")

We hope it will sparkle curiosity about AI in general and Reinforcement Learning in particular and make the player want to read the source code.

## INSTALLATION

Binaries can be downloaded :
For windows : https://www.dropbox.com/s/n9nz5p9ua0ne9t8/Outsmart.exe?dl=0
For Mac : http://rdklein.fr/launch.zip

There is no Linux package at the moment, although it will run from source.

You need Python >=3.4 (it may work with 3.3 but is untested), avbin (https://code.google.com/p/avbin/downloads/detail?name=avbin-linux-x86-64-7.tar.gz), pyglet and numpy (pip install pyglet numpy).

Then you can run `./launch.py`

## GAMEPLAY

Follow the in-game tutorial. Keep in mind that the robots can only "see" the tiles immediately next to them.

## CONTEST

This program is an entry to gamedevfort.com's game making contest :

http://contest.gamedevfort.com/submission/432/

We would appreciate a vote :)


## REINFORCEMENT LEARNING

We use the Q-learning algorithm. The RL code in rl.py is not technically tied to the game and can be used in any other application (but license-wise it is under the AGPL as the rest of the game).

## BACKSTORY

The game is set in a dystopian close future, where autonomous, self-replicating combat drones have been released in the wild.

As the intro video explains :

>For years, lone voices in the Artificial Intelligence research community had try to warn the general public about the global threat to humanity that an Artificial General Intelligence would be.

>A smarter-than-humans AI, they said, would be unstoppable. In a thought experiment framed by those pioneers of Artificial General Intelligence Safety, such a smart AI would lure a human operator to let it out of the "box" it is confined in, only to wreck havoc on the world.

>The top priority, those researchers warned, should be to research ways to implement a friendly AI, whose goals are aligned with those of humanity. They set out to find those ways.

>All this was, of course, a load of bullshit.

>While the fear mongerers spent their time writing Harry Potter fanfiction or speculating idly on non-sensical thought experiments on ill-defined concepts, as relevant as Aristotle warning his fellow Greeks about the danger of nuclear reactor meltdowns after coining the word "atom", the actual AI researchers were busy trying to secure funding for actual AI research.

>The trouble came from the source of this funding. Because selling ads, making toys and video games or helping medical science, physics or public engineering get forward is not lucrative enough to secure a steady stream of research funding, true fundamental research money came from the military.

>Ask yourself, would you rather wrestle with Stephen Hawking, or a turret from Portal ?

>In retrospect, it was kind of obvious. Not everybody missed it. Great people asked for a ban on AI-assisted warfare. Giving dumb AIs guns, camouflage, self-repair capabilities and energetic autonomy was a really dangerous idea.

>Oh, and we did worse than that, we gave those dumb AIs ways to duplicate themselves, as long as they can find the resources for that.

>Also, we asked them to optimize for the collection of said resources.

>Your job, now, is to outsmart the dumb AIs we have foolishly released into the wild. You must use their craving for resources to somehow lure them into disabling traps, and do so before they gather enough resources to grow in numbers.

## BUGS

Plenty of them undiscovered. Issues and pull requests are welcomed :)

## HISTORY

Proof of concept was done on Jul 11, 2015, but actual development started on Jul 30, 2015.

## AUTHORS

Denis Baheux and Edouard Klein

## LICENSE

AGPL

## SEE ALSO

The box experiment : http://rationalwiki.org/wiki/AI-box_experiment and
https://xkcd.com/1450/

Ban on AI warfare (signed by Russel (who wrote the seminal paper in the domain of Inverse Reinforcement Learning))
and Stephen Hawking) :
http://futureoflife.org/AI/open_letter_autonomous_weapons

Harry potter fanfiction : http://hpmor.com/

Military funding, e.g. the DARPA challenge :
https://www.youtube.com/watch?v=bwa8m8VwhWU

Proponents of the rogue general AI hypothesis (with which the authors disagree) :
https://intelligence.org/2015/07/27/miris-approach/
