# Script
## Intro
So... I'm making a game for the PS2 in 2024. But why?
Well, I'm a simple man, and I only need simple reasons you see.

`Illustrate situation`
As I was talking with a coworker of retro video games (probably a Final Fantasy or a Dragon Quest game), I wondered if it would still be possible to make a game for one of the first modern consoles.

`Show illustrations of explanation`
What do I mean by modern consoles ? Well, let's start from the beginning. Nowadays, you can make a game for, let's say, any console released before the Gamecube and the PS1 without too much trouble. As far as I know, the Software Development Kit (which I'll call SDK from now on) released with those consoles was... Well, pretty light, if not non-existent. You basically had to write everything yourselves, so even If it might take a while, it's basically just: I put X value in X register, and I get X pixel on the screen.

`Show modern consoles hardware diagrams and illustrations`
However, consoles released after that, such as the Xbox or the PS2, are much more complex machines, that offers more capabilities than their predecessors. That mean that in order to write a game for the PS2, I would need to write a large library to handle basic functionalities, which would take a while, would be difficult, and expensive. So, in order to have more than 2 obscure games released on their consoles, console manufacturers wrote those library for us already, those so-called SDKs.

## The PS2

Knowing the closed nature of video game development at the time, I just simply wanted to see if those were available to the public, more than 20 years after. And to my surprise...

`Show ps2sdk link`
Yup. There is an SDK online. Actually, there is a whole organization with plenty of repositories containing tools for programming on the PS2. Now, those are actually open-source, and are retro-engineered to mimic what the real PS2 SDK was doing, but that's already a whole lot!

And so I thought to myself, why not try it? Why shouldn't I try to make a game using the PS2 SDK? After all, it's in C++, and I love C++! So I started digging... And digging... And digging...
Until I came to a realization. I was not up to the task, at all. I'll admit I didn't try for too long, but the sheer lack of documentation on how to get started was enough to get me discouraged.

So, for a few days, I just went on to something else... Until a new question came to my mind:

"Is there a modern game engine for the PS2 ?"

And so I put my detective talent at work again, and quickly find that yes... But also no.
So, there's this repository called Tyra. Now, Tyra claims to be a game engine for the PS2. But I'd actually rather call it a high level library for the PS2 SDK.

`Show sample of Tyra`
It actually simplifies the code that we would need to write in order to load and display a 3D model for example, or to handle light, and sprites rendering. But, it doesn't includes the other required parts of a game engine:
- There are no graphical editor for a level for example, which is understandable as it would take some effort to make one for, arguably, not much, as nobody make games for the PS2 anymore (I mean except me of course)
- And more importantly, there isn't an ECS or something similar to handle entities. This mean that without one, we would have to handle each entity in our game by hand in the code. Which, for a simple game is fine, such as a Duck hunt clone for example, but come on, this isn't 1984 anymore, we can make so much more with the PS2.

## My game

So, the first step of my project was very clear: I need to upgrade Tyra with a couple of features before I could start working on making a game.

But... What game do I actually want to make? Let's see... I learned about Tyra through talking with a coworker about games such as Final Fantasy and Dragon Quest...
Ha-ha, I know! I'll just let my coworker decide on what the game should be! I actually suck at Game Design, and I'll already have my hands busy with writing and upgrading Tyra, so trust me, it's better if someone else take care of that part of the game.

`Illustrate with screenshot from DQ 4-5-6`
But since we don't want to make the game too complex, we'll probably go with a mix of 3D and 2D, just like what we can see in the Dragon Quest remakes for the Nintendo DS. And a twin stick shooter too, cause why not.

So anyway, the plan:
- I need to add some utils functions. Tyra is nice, and helps a ton, but it still misses some basic functions that I will definitely need in the game, such as drawing text. So I'll have to take care of that first.
- Then, the backbone of the game will be the way it handles entities. I already have an idea, and I'll probably start by integrating ENTT into Tyra. For those that don't know, ENTT is a well-known ECS library that is used in pretty big engine, such as the one that Mojang uses for example. Yup, Minecraft (not the Java edition) uses ENTT as an ECS, so I don't think it will have much trouble with my game.
- And then... Well, we'll see. I already have a few idea, such as handling map loading, so that I can start prototyping as soon as possible, but all of that can wait for now, I already have quite a lot on my hand.

...Actually, I lied. I have already done the first 2 steps that I just talked about. Basic text drawing and ENTT are already implemented in the game, so it's not like the project hasn't even started yet.

## Outro

`Show clip of me struggling on something`
And that makes the perfect transition for me to tell you about my Twitch account, where I stream myself struggling on basic problems on the project.

If you want to see more of what I do, or of the evolution of the project before anyone else, you can find me on Twitch at Feisar_labs. I've actually already streamed a few times. The VODs are online on my youtube channel, if you missed those (which you probably did, but I'm not blaming you for that).

If you want to track the progress I make, you can find the repository in the description of the video. I will probably update the code daily, or at least a couple times a week.
If some of you guys are living in France, in Brittany to be precise, I will be giving a short "tool-in-action" about Tyra (in French, sorry !) at the Breizhcamp, in June.

If you wonder why I made this video when I don't have a lot to show yet, it's because I find this project very interesting, and I want to keep track of what I've done, so you can expect other videos about this project to release in the future, talking about what I've achieved.

Anyway, that's it for today, if you liked this video and want to see me more, let me know by liking this video and subscribing to this channel. If you have questions about the project, please comment under that video or come ask me during a stream, I'll be happy to answer you!

Thank you for watching. Feisar, out.