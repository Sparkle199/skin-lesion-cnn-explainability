# The Simple Version: What This Whole Project Was About

*A plain-English companion to Chapters 1 to 5, for talking someone through the project without the jargon. Every hard word gets a simple explanation first, with the real technical term in brackets right after, so you can still use the correct words if someone asks.*

---

## Chapter 1: Why We Even Did This

Skin cancer is a sickness where skin cells start growing the wrong way and won't stop (cancer). Doctors take close-up photos of skin spots (dermoscopic images) to figure out if a spot is dangerous or not. The tricky part is that some dangerous spots look a lot like harmless ones, even to trained doctors, and catching the dangerous ones early makes a huge difference: caught early, almost everyone survives; caught late, most people don't.

So the idea was: what if we teach a computer to look at thousands of these photos and learn to guess what each spot is? The type of computer program that's good at "looking" at pictures and learning patterns from them is called a neural network built for images (a convolutional neural network, or CNN for short). We didn't just build one of these programs, we built three different ones, kind of like three different students who all study the same textbook but think a bit differently, and we wanted to see which "student" was best.

But there was a second question we cared about just as much: if the computer says "this spot is dangerous," can it show its work? A computer that just says "trust me" isn't very useful to a doctor. So we also tested tools that make the computer point at the part of the photo it was actually looking at when it made its guess (explainable AI, or XAI). And a third question: does the computer work equally well for everyone, including people with darker skin, since most of the practice photos it learns from happen to be from lighter-skinned patients?

Important: none of this was meant to become a real medical tool. It's an experiment to answer these three questions honestly, not a product.

## Chapter 2: What Other People Had Already Tried

Before starting, we looked at what other researchers had already done (a literature review). Some of what we found was impressive: a few studies had already built computer programs that were as good as, or even better than, real dermatologists at answering simple yes-or-no questions like "is this dangerous or not" (a binary classification task).

But we noticed some gaps. Nobody had fairly compared our exact three "student" programs (ResNet-50, EfficientNetB4, and VGG-16, the names of the three CNN designs we used) on the hardest version of the test, where the computer has to pick from seven different possible answers instead of just two (a seven-class classification task), using the full, real, messy set of photos rather than a cleaned-up, easier version. Some earlier studies had quietly thrown away thousands of the "boring, common" photos to make their numbers look better, which isn't a fair test of the real world.

We also noticed that when researchers did add a "show your work" feature to their computer, almost nobody actually checked whether the computer was pointing at the right spot. They just showed a picture with some highlighting and assumed it looked reasonable. And almost no one had tested whether their computer worked fairly across different skin tones.

So we set out to fix all three of those gaps at once.

## Chapter 3: How We Set Up the Experiment

**The photos.** We used a big public collection of about 10,000 skin photos with known answers already checked by doctors (the HAM10000 dataset), plus a second, smaller collection of 656 photos specifically chosen to include a much wider range of skin tones (the DDI dataset, for Diverse Dermatology Images). We also grabbed a completely separate set of 1,511 photos that none of our computers had ever seen before, just to double check our results weren't a fluke (an independent test set called ISIC2018).

**The two games.** We set our three computer "students" two different games to play. Game one: look at a photo and pick one of seven specific skin conditions (the seven-class task). Game two: simpler, just decide if a spot is dangerous or not-dangerous (the binary malignant-or-benign task), and this is the game where we added in the extra diverse-skin-tone photos.

**Making it a fair test.** All three computer students studied from the exact same set of practice photos, with the exact same rules, so that if one did better than another, it's really because of how that "student" is built, not because it got easier questions. We also made sure a photo of the same skin spot never accidentally showed up in both the "practice" pile and the "test" pile, since that would be like letting a student see the exam questions during homework (this is called lesion-level splitting, to avoid data leakage).

**Testing fairness across skin tones.** Since the DDI photos have skin-tone labels, we tried three different ways of using them: show the computer zero diverse photos and just see how badly it does on its own (zero-shot), teach it the regular way first and then give it some extra diverse-photo practice afterward (fine-tuning), or mix the diverse photos in right from the very start of practice (joint training).

**Checking the computer's "pointing."** For the explainability check, we used two different tools. One works like a heat map showing which part of the photo lit up the computer's attention the most (Grad-CAM). The other works by testing what happens to the computer's answer if you hide different parts of the photo, a bit like a guessing game (SHAP). Then, since HAM10000 photos come with an outline already drawn around the actual skin spot by a doctor (a segmentation mask), we could measure, with a real number, how much the computer's "pointing" overlapped with where the spot actually was (using overlap scores called IoU and Dice).

**Being careful with small tests.** Early on, we noticed something important: if you only test on a small number of photos, say 15 or 30, the results can be misleading, kind of like judging how good a basketball player is from watching them take only 5 shots. So every time we got an interesting result from a small test, we didn't trust it until we reran it on a much bigger pile of photos (up to 500) and checked the result held up with proper statistics (a confidence interval).

## Chapter 4: What We Actually Found

**Which student did best at the hard game?** On the seven-choice game, two of our three computer students, ResNet-50 and VGG-16, finished neck and neck, both getting roughly two out of every three photos right. The third, EfficientNetB4, trailed a bit behind on basically every measurement we used.

**Which fairness strategy worked best?** Mixing the diverse skin-tone photos in from the very start of training (joint training) beat both of the other two approaches by a wide margin, for all three computer students. Just testing a computer that had never seen a diverse photo (zero-shot) was rough: it went from being right about 80 to 87 times out of 100 dangerous cases on its normal practice photos, down to being right only 12 to 48 times out of 100 dangerous cases on the diverse photos. That's a big, real fairness problem, and it shows exactly why we bothered testing it this way.

**Was it fair across every skin tone?** Here's a twist: once we used the "mix from the start" strategy, none of the three computer students showed a simple pattern of "worse on darker skin." One of them was actually best on the darkest skin-tone group in our test. This doesn't mean the fairness problem is solved (our diverse-photo groups were still fairly small, so we can't be too confident), but it does mean the story is more complicated than a straight line from lighter to darker skin.

**Which "pointing" method was more trustworthy?** This is where the small-test-versus-big-test lesson really paid off. In our very first quick check, it looked like one method (SHAP) was clearly better for EfficientNetB4. But once we reran it properly on hundreds of photos and did the statistics, that difference disappeared entirely, meaning the two methods are basically tied for that computer student. For the other two students, the differences we found were real and held up: ResNet-50's heat map method (Grad-CAM) was genuinely more trustworthy for that computer, while VGG-16's guessing-game method (SHAP) was genuinely more trustworthy for that one, and this held up even at the biggest sample size we tried.

## Chapter 5: What It All Means, and What's Next

**So what did we learn?** Two of our three computer students (ResNet-50 and VGG-16) are roughly equally good at the hard seven-choice game, and the third (EfficientNetB4) is consistently a bit behind. Training with diverse skin-tone photos mixed in from the start is clearly the best way to make the fairness gap smaller, though it doesn't erase it completely. And which "explain yourself" method works best genuinely depends on which computer student you're asking, it's not a one-size-fits-all answer, which is itself a useful and previously-untested finding.

**What are the honest limits?** We only trained each computer once per game, so we can't yet say for certain how much luck was involved (that would need repeating each experiment several times, called checking for variance). Our diverse-skin-tone test groups were small, so those specific fairness numbers should be treated as a first clue, not the final word. Nobody real, like an actual dermatologist, has yet looked at the computer's "pointing" and confirmed it actually makes clinical sense, only that it overlaps with the outline a computer algorithm drew. And again: this is a research experiment, not a real medical tool, and it should never be treated as one.

**What should happen next?** Get real dermatologists to look at the computer's explanations and say whether they actually make sense (clinical validation), not just whether they overlap with an outline. Repeat every experiment a few more times with different random starting points to make sure the results aren't just luck (multi-seed testing). Get a bigger collection of diverse-skin-tone photos so the fairness numbers can be trusted with more confidence. And figure out why VGG-16 in particular seems to prefer the guessing-game explanation method, since right now we know that it does, but not exactly why.
