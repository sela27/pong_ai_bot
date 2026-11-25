# וריאנט האלגוריתם: DQN מהמאמר או עם שיפורים

מחקר לטיקט [#3](https://github.com/sela27/pong_ai_bot/issues/3) במפה [#1](https://github.com/sela27/pong_ai_bot/issues/1).
המונחים (Agent, Opponent, Game, Point, Observation, Reward, Score margin, Evaluation) מוגדרים ב-`GLOSSARY.md`.
כל טענה מסומנת במספר מקור בסוגריים מרובעים, והרשימה המלאה בסוף. מקומות שבהם יש חישוב שלי ולא ציטוט מסומנים במילה **חישוב**, ומקומות לא ודאיים מסומנים במילה **לא ודאי**.

## תקציר

- היעד שלנו (Score margin ממוצע של 18 נקודות לפחות לטובת ה-Agent) הוא בערך הרמה של ה-DQN המקורי: במאמר של 2015 ה-DQN קיבל ב-Pong ממוצע של 18.9 (סטיית תקן 1.3) [1].
- ב-Pong כל הווריאנטים מגיעים כמעט לתקרה (בין 19.5 ל-21 בטבלאות של Dueling ושל Rainbow) [4][6]. לכן השיפורים קונים בעיקר **מהירות אימון**, לא ציון סופי גבוה יותר.
- מימושי ייחוס מודרניים של DQN פשוט, עם Adam במקום RMSProp, מגיעים ב-ALE Pong לציון של בערך 20 אחרי 10 מיליון צעדי Agent [14][16].
- **ההמלצה:** להתחיל מה-DQN של Nature 2015 (replay buffer, target network, אותה רשת קונבולוציה, 4 פריימים מוערמים), עם שני שינויים קלים ומבוססים: Adam במקום RMSProp, ו-target של Double DQN (שינוי של שורה אחת). לא להתחיל עם Rainbow. אם האימון איטי מדי, להוסיף קודם n-step returns ואחר כך Prioritized Replay. הפירוט בסעיף "המלצה".

## 1. מילון קצר: frame מול צעד Agent

המונח "frame" משמש בספרות בשתי משמעויות, וזה משנה כל מספר בהמשך:

- **פריים של המשחק (emulator frame):** ב-Atari יש 60 כאלה בשנייה [1].
- **צעד Agent (agent step):** החלטה אחת של ה-Agent. עם frame skip של k=4, כל צעד Agent מכסה 4 פריימים של המשחק [1].

ב-Nature כתוב שהאימון היה "50 million frames (that is, around 38 days of game experience)" [1]. **חישוב:** ‏50 מיליון כפול 4 פריימים, חלקי 60 בשנייה, זה בערך 38.6 ימים. כלומר ה-"frames" של Nature הם בפועל צעדי Agent. גם Dopamine מתרגמים את הפרמטרים של Nature כך: התחלת למידה אחרי 200,000 פריימים, ו-ε שיורד לאורך 4,000,000 פריימים [9]. ‏Machado ושות' מציינים שהספרות לא מדווחת על זה באופן עקבי, וממליצים לספור את הפריימים המדולגים [7].

במסמך הזה "צעד" פירושו צעד Agent, ו"פריים" פירושו פריים של המשחק, אלא אם כתוב אחרת.

## 2. מה בדיוק עשה ה-DQN של Nature 2015

### 2.1 עיבוד התמונה (preprocessing)

- **מקסימום על שני פריימים:** לכל פיקסל לוקחים את הערך המקסימלי מהפריים הנוכחי ומהפריים הקודם. הסיבה: ב-Atari יש אובייקטים שמופיעים רק בפריימים זוגיים או רק באי-זוגיים (flicker), בגלל מגבלת ה-sprites של המכונה [1].
- **גווני אפור:** מחלצים את ערוץ ה-Y (luminance) מה-RGB, ומקטינים ל-84×84 [1].
- **ערימת פריימים:** ה-Observation שנכנס לרשת הוא 4 הפריימים האחרונים (m=4). לפי המאמר האלגוריתם עמיד גם ל-3 או ל-5 [1].
- **הסיבה לערימה:** ממסך בודד אי אפשר להבין את המצב המלא (perceptual aliasing), למשל לאן הכדור נע [1][2].
- **הגרסה של 2013 הייתה קצת אחרת:** גווני אפור, הקטנה ל-110×84, ואז חיתוך של 84×84 מאזור המשחק. החיתוך נדרש רק כי מימוש הקונבולוציה שלהם דרש קלט ריבועי [2].

### 2.2 מבנה הרשת

לפי Nature [1]:

| שכבה | פירוט |
|---|---|
| קלט | `84×84×4` |
| קונבולוציה 1 | ‏32 פילטרים `8×8`, ‏stride 4, ‏ReLU |
| קונבולוציה 2 | ‏64 פילטרים `4×4`, ‏stride 2, ‏ReLU |
| קונבולוציה 3 | ‏64 פילטרים `3×3`, ‏stride 1, ‏ReLU |
| fully connected | ‏512 יחידות ReLU |
| פלט | שכבה לינארית, יציאה אחת לכל פעולה חוקית |

- ברשת יש יציאה לכל פעולה, כך שמעבר אחד ברשת נותן את ה-Q של כל הפעולות [1].
- ברשת של 2013 היו רק שתי שכבות קונבולוציה (16 ו-32 פילטרים) ו-256 יחידות [2].
- **חישוב:** עם 3 פעולות יש לרשת של Nature בערך 1.69 מיליון פרמטרים, רובם בשכבת ה-512 (‏`3136×512`). ‏CleanRL משתמשים באותם מימדים בדיוק: `Linear(3136, 512)` [15].

### 2.3 האלגוריתם

- ‏**Experience replay:** שומרים את המעברים `(s, a, r, s')` בזיכרון ודוגמים מהם minibatch באופן אחיד ואקראי. זה שובר את המתאם בין צעדים רצופים ומחליק את התפלגות הנתונים [1].
- ‏**Target network:** כל C עדכונים מעתיקים את הרשת לרשת נפרדת, שמחשבת את ה-target. זה מוסיף השהיה בין עדכון לבין השפעתו על ה-target, ומקטין תנודות והתבדרות [1].
- ‏**Target במצב סופי:** כשה-Game נגמר בצעד הבא, ה-target הוא רק `r`, בלי bootstrap [1].
- ‏**Error clipping:** במאמר חתכו את איבר השגיאה לטווח `[-1, 1]`. זה שקול ל-loss ריבועי בתוך הטווח ול-loss של ערך מוחלט מחוצה לו [1], כלומר Huber loss. ‏OpenAI מזהירים שקל לפרש את זה לא נכון ולחתוך את ה-loss עצמו, ואז הגרדיאנט מתאפס בדיוק במעברים עם השגיאה הגדולה [22].
- ‏**Reward clipping:** כל Reward חיובי הפך ל-1 וכל שלילי ל-`-1`. הסיבה הייתה שסקאלת הניקוד שונה מאוד בין משחקים [1].
- **סוף episode באיבוד חיים:** במשחקים עם חיים, איבוד חיים סימן סוף episode באימון [1]. ב-Pong אין חיים, וזה לא רלוונטי לנו.
- ‏**Optimizer:** ‏RMSProp עם minibatch של 32 [1].

### 2.4 אימון והערכה

- **אורך האימון:** ‏50 מיליון צעדים (200 מיליון פריימים), עם replay memory של מיליון המעברים האחרונים [1].
- ‏**Frame skip:** ה-Agent בוחר פעולה כל k=4 פריימים, והפעולה חוזרת על עצמה בפריימים שבאמצע [1].
- **ה-Hyperparameters נבחרו בחיפוש לא פורמלי** על חמישה משחקים, ו-Pong אחד מהם [1].
- **פרוטוקול ההערכה:** ‏30 משחקים, כל אחד עד 5 דקות, עם עד 30 פעולות "no-op" אקראיות בהתחלה, ועם ε=0.05. המטרה הייתה להקטין את הסיכוי ל-overfitting בהערכה [1].
- **הציון ב-Pong** [1]:

| שחקן | ציון |
|---|---|
| ‏Random | `-20.7` |
| אדם (בודק משחקים מקצועי) | `9.3` |
| ‏DQN | `18.9 (±1.3)` |

- ב-2013 דווח ב-Pong ממוצע של 20 ומשחק הכי טוב של 21, אחרי 10 מיליון "frames" ועם ε=0.05 [2].
- **למה replay ו-target network חשובים:** ב-Nature כיבו כל אחד מהם בנפרד (אחרי 10 מיליון frames, בלי Pong בטבלה). ב-Breakout, למשל, הציון ירד מ-316.8 ל-10.2 בלי replay, ול-3.2 בלי שניהם [1].

### 2.5 טבלת ברירות המחדל של 2015 (לעיון בלבד)

הטבלה מועתקת מ-Extended Data Table 1 [1]. היא לא המלצה לערכים שלנו: כיוון ה-hyperparameters נשאר פתוח במפה.

| Hyperparameter | ערך | תיאור במאמר |
|---|---|---|
| minibatch size | 32 | מספר הדוגמאות בכל עדכון SGD |
| replay memory size | 1,000,000 | העדכונים נדגמים מהמספר הזה של ה-"frames" האחרונים |
| agent history length | 4 | מספר הפריימים האחרונים שנכנסים לרשת |
| target network update frequency | 10,000 | נמדד ב"parameter updates" (זה הפרמטר C) |
| discount factor γ | 0.99 | |
| action repeat | 4 | ה-Agent רואה רק כל פריים רביעי |
| update frequency | 4 | מספר הפעולות בין שני עדכוני SGD |
| learning rate | 0.00025 | של RMSProp |
| gradient momentum | 0.95 | של RMSProp |
| squared gradient momentum | 0.95 | של RMSProp |
| min squared gradient | 0.01 | קבוע במכנה של RMSProp |
| initial exploration ε | 1 | |
| final exploration ε | 0.1 | |
| final exploration frame | 1,000,000 | ε יורד לינארית לאורך המספר הזה של "frames" |
| replay start size | 50,000 | מדיניות אקראית רצה כך כדי למלא את הזיכרון לפני הלמידה |
| no-op max | 30 | מקסימום פעולות "לא לעשות כלום" בתחילת episode |

**לא ודאי:** לפי הטבלה, ה-target network מתעדכן כל 10,000 "parameter updates" [1]. ‏Dopamine מתרגמים את זה ל-10,000 צעדי Agent (‏40,000 פריימים) [8][9]. מכיוון שיש עדכון כל 4 צעדים, שתי הקריאות שונות פי 4. בכל מקרה זה עניין של כיוון hyperparameters, ולא של בחירת הווריאנט.

## 3. מה מוסיף כל שיפור, ומה זה נותן ב-Pong

### ‏3.1 Double DQN

- **מה זה:** ב-target, הרשת הנוכחית בוחרת את הפעולה הכי טובה, ו-target network מעריך את הערך שלה: `y = r + γ·Q_target(s', argmax_a Q_online(s', a))` [3].
- **מה זה פותר:** הערכת-יתר (overestimation) של ערכי Q. ב-DQN נמצאה הערכת-יתר בכל 49 משחקי Atari שנבדקו [3].
- **מחיר:** תקורה חישובית מינימלית, בלי רשתות או פרמטרים נוספים [3].
- **ב-Pong:** עם no-op starts הציון עלה מ-18.90 ל-21.00. עם human starts הוא עלה מ-16.2 ל-17.7, ועם גרסה מכוונת (tuned) ל-19.1 [3].
- **הסתייגות של המחברים עצמם:** הערכת-יתר לא תמיד פוגעת, ו-DQN מגיע למשחק אופטימלי ב-Pong למרות שהוא מעריך-יתר [3].
- **ב-Rainbow:** הסרת Double מ-Rainbow לא שינתה הרבה את החציון. לפעמים הוא עזר ולפעמים הזיק, תלוי במשחק [6].

### ‏3.2 Dueling network

- **מה זה:** אחרי שכבות הקונבולוציה הרשת מתפצלת לשני ענפים של 512 יחידות. ענף אחד מעריך את V(s), והשני את A(s,a) לכל פעולה. מחברים אותם כך: `Q = V + (A − mean(A))` [4].
- **איפה זה עוזר:** כשיש הרבה פעולות עם ערך דומה. הפער גדל עם מספר הפעולות [4]. ב-Pong יש רק 3 פעולות [4], כך שהיתרון הזה קטן כאן.
- **שינויים באימון:** הגרדיאנט שנכנס לשכבת הקונבולוציה האחרונה מוכפל ב-`1/√2`, ו-gradient norm נחתך ל-10 [4].
- **ב-Pong (no-op):** ‏DQN קיבל 19.5, ‏DDQN קיבל 20.9, ו-Dueling קיבל 21.0 [4]. עם human starts הסדר מתהפך חלקית: ‏DDQN קיבל 19.1, ו-Dueling קיבל 18.8 [4].
- **ב-Rainbow:** לא נמצא הבדל מובהק כשהסירו את Dueling מ-Rainbow המלא [6].

### ‏3.3 Prioritized Experience Replay (PER)

- **מה זה:** דוגמים מעברים מה-replay לפי גודל השגיאה שלהם (TD error), ולא באופן אחיד [5].
- **מחיר במורכבות:**
  - מבנה נתונים של sum-tree, עם עדכון ודגימה ב-`O(log N)` [5].
  - משקלי importance sampling, עם מעריך β שעולה לינארית עד 1 [5].
  - שני hyperparameters חדשים: α=0.6 ו-β₀=0.4 בווריאנט ה-proportional [5].
  - קצב הלמידה הוקטן פי 4 [5].
- **מה זה נותן:** "learning is twice as fast" במצטבר. הווריאנטים עם PER הגיעו לביצועים החציוניים של Double DQN אחרי 47% ו-38% מזמן האימון [5]. זה נתון חציוני על כל המשחקים, לא נתון של Pong.
- **ב-Pong (human starts, ציון מנורמל):** ‏DQN קיבל 102%. כל השאר (DQN עם PER, ‏Double DQN, ו-Double DQN עם PER) קיבלו בין 110% ל-111% [5].
- **ב-Rainbow:** ‏PER ו-n-step הם "the two most crucial components" [6].

### ‏3.4 n-step returns

- **מה זה:** ה-target מסכם n תגמולים אמיתיים לפני ה-bootstrap, במקום תגמול אחד. ב-Rainbow השתמשו ב-n=3 [6].
- **מה זה נותן:** לפי ה-ablation של Rainbow, זה אחד משני הרכיבים החשובים ביותר [6]. בטבלת הדירוג של Revisiting Rainbow, ‏n-step מופיע כרכיב המוביל בכמה סביבות, כולל Breakout, ‏Seaquest ו-SpaceInvaders בגרסת MinAtar [13]. **לא ודאי:** המאמר לא מסביר בבירור איך חושב הדירוג.
- **מחיר:** **חישוב/הערכה שלי:** שינוי קטן יחסית בבניית ה-target וב-replay. אין לו מבנה נתונים חדש כמו ב-PER.

### ‏3.5 Distributional (C51) ו-Noisy Nets

- ‏**Distributional (C51):** הרשת לומדת התפלגות של התשואה עם 51 נקודות (atoms) בטווח `[-10, 10]`, ולא ממוצע [6].
- ‏**Noisy Nets:** אקראיות נלמדת בשכבות של הרשת, במקום ε-greedy [6].
- **ב-Rainbow:** ‏Distributional היה פחות חשוב מ-PER ומ-n-step, אבל חשוב יותר מ-Dueling ומ-Double. הסרת Noisy Nets הורידה את החציון [6].
- **מסקנה ללימוד:** **הערכה שלי:** אלה שני הרכיבים הכי קשים להבנה ולדיבוג, ובפרויקט לימודי הם לא שווים את המחיר בשלב הראשון.

### ‏3.6 Rainbow (כל השישה יחד)

- **מה זה נותן:** הוא מגיע לביצועים הסופיים של DQN אחרי 7 מיליון פריימים, בחציון על כל המשחקים [6]. ‏7 מיליון הפריימים האלה לקחו פחות מ-10 שעות, וריצה מלאה של 200 מיליון פריימים לקחה בערך 10 ימים על GPU יחיד [6].
- **ב-Pong:** ‏Rainbow קיבל 20.9 (no-op) ו-19.0 (human starts) [6]. זה לא יותר טוב מ-Dueling DDQN (‏21.0 ו-18.8) [6], כי Pong כבר ברוויה.
- **ה-hyperparameters שלו** [6]:
  - ‏Adam עם lr=0.0000625 ו-eps=1.5e-4.
  - ‏n=3.
  - ‏target network כל 32K פריימים.
  - התחלת למידה אחרי 80K פריימים.
  - ‏51 atoms.
  - ‏priority exponent 0.5.
  - ‏β שעולה מ-0.4 ל-1.0.

### 3.7 סיכום: רכיב, תועלת ב-Pong, ומחיר

| רכיב | ציון Pong (no-op) | השפעה כללית לפי המקורות | מחיר במורכבות |
|---|---|---|---|
| DQN (Nature) | 18.9 [1], ובהרצה חוזרת 19.5 [4] | הבסיס | בסיס |
| + Double | 21.0 [3], ‏20.9 [4] | הערכת-יתר קטנה יותר; בחציון של Rainbow ההשפעה מוגבלת [6] | שורה אחת, בלי פרמטרים [3] |
| + Dueling | 21.0 [4] | אין הבדל מובהק כשמסירים מ-Rainbow [6] | ראש רשת כפול |
| + PER | 20.7 (Prior. DDQN) [6] | בערך פי 2 מהר יותר בחציון [5] | ‏sum-tree, משקלי IS, ‏2 פרמטרים [5] |
| + n-step | אין נתון Pong נפרד | אחד משני הרכיבים החשובים ב-Rainbow [6] | נמוך (**הערכה**) |
| Rainbow | 20.9 [6] | ביצועי DQN הסופיים אחרי 7M פריימים (חציון) [6] | גבוה: שישה רכיבים |

## 4. כמה אימון צריך כדי להגיע ל-18 ב-Pong

### 4.1 נתונים ממימושי ייחוס (כולם על ALE Pong, לא על Pong שלנו)

| מימוש | שינויים עיקריים מול Nature | צעדים | ציון Pong |
|---|---|---|---|
| Nature DQN [1] | (המקור) | 50M | `18.9 ± 1.3` |
| CleanRL `dqn_atari.py` [14][15] | ‏Adam ‏lr=1e-4; ‏MSE; ‏ε סופי 0.01; ‏target כל 1000 צעדים | 10M | `20.25 ± 0.41` |
| CleanRL `dqn_atari_jax.py` [14] | כמו הקודם, 3 seeds | 10M | `20.43 ± 0.34` |
| RL Zoo (SB3) DQN [16][17][19] | ‏Adam ‏lr=1e-4; ‏Huber; ‏gradient clipping ל-10; ‏ε סופי 0.01 | 10M | `20.602 ± 0.613` (88 episodes) |

- ‏CleanRL כותבים שהמימוש שלהם "scores the same or higher than (Mnih et al., 2015) in 10M steps" [14].
- ‏SB3 מציינים שזו ריצה יחידה ולא השוואה כמותית קפדנית. ב-Atari ההערכה שם לא דטרמיניסטית [16].

### 4.2 מתי חוצים את 18: נתוני Dopamine

‏Dopamine מפרסמים עקומות אימון של Pong לכל agent: חמש ריצות (seeds), ו-200 איטרציות של 250,000 צעדים כל אחת [10][11]. התנאים שם קשים יותר משלנו:

- ‏sticky actions בהסתברות 0.25 [11].
- הציון הוא ציון האימון עם ε=0.01 [11].

**חישוב:** קראתי את `pong.json` [10], מיצעתי את חמש הריצות בכל איטרציה, ובדקתי באיזו איטרציה הממוצע עבר לראשונה כל סף. איטרציה `i` מסתיימת אחרי `(i+1)×250K` צעדים.

| Agent ב-Dopamine | ממוצע 15 | ממוצע 18 | מקסימום בתוך 50M צעדים |
|---|---|---|---|
| DQN, ‏RMSProp + Huber (בסגנון Nature) | ~25M צעדים | **לא הגיע** | 16.6 |
| DQN, ‏Adam + MSE | ~4.5M צעדים | ~10M צעדים | 19.8 |
| C51 | ~6.25M צעדים | ~13.25M צעדים | 19.5 |
| "Rainbow" של Dopamine (‏C51 + 3-step + PER) | ~2.5M צעדים | ~4M צעדים | 20.2 |
| IQN | ~2.5M צעדים | ~4.5M צעדים | 20.3 |

- ה-"Rainbow" של Dopamine כולל רק שלושה רכיבים: n-step, ‏PER ו-distributional. אין בו Double, ‏Dueling ו-Noisy Nets [8].
- ב-DQN עם RMSProp הפיזור בין ה-seeds בסוף הוא בין 13.0 ל-19.0. ב-DQN עם Adam הפיזור הוא בין 19.5 ל-20.1 [10].
- ‏Dopamine כותבים שהם עברו ל-Adam ב-DQN כי "using Adam yields improved training performance" [11].
- ‏Revisiting Rainbow מצאו על כל משחקי Atari ש-"Adam+MSE is a superior combination than RMSProp+Huber" [13].
- ‏Dopamine כותבים גם ש-DQN "performs much better" עם 1% פעולות אקראיות באימון, במקום 10% כמו ב-Nature [9].

### 4.3 מה זה אומר לנו (הערכה)

- **מספר הצעדים:** ב-ALE Pong, ‏DQN פשוט עם Adam מגיע לממוצע של 18 בערך אחרי 10M צעדים, בתנאים הקשים של Dopamine [10]. בתנאים הרגילים (בלי sticky actions) הוא מגיע ל-20 בסוף 10M צעדים [14][16].
- ‏**Rainbow חלקי:** מגיע לשם בערך פי 2.5 מהר יותר בצעדים [10]. אבל כל צעד שלו יקר יותר וקשה יותר לדיבוג.
- ‏**Evaluation שלנו שונה:** היא בלי אקראיות (ε=0), ואולי בלי sticky actions. לכן **סביר** שנעבור את הסף מוקדם יותר מהטבלה שבסעיף 4.2. **לא ודאי:** אין לי מקור שמודד את זה ישירות.
- **השוואה ל-Nature לא מדויקת:** ההערכה שם הייתה עם ε=0.05 ועד 5 דקות למשחק [1]. ייתכן שחלק מהמשחקים נחתכו לפני 21 Points, ולכן 18.9 הוא לא בדיוק Score margin של Game מלא כמו אצלנו (**לא ודאי**).

## 5. מה זה אומר על ה-Quadro P2200

### 5.1 זיכרון GPU (‏5 GB): לא צוואר הבקבוק

- **חישוב:** לרשת של Nature יש כ-1.69M פרמטרים, שהם כ-6.7MB ב-float32. גם עם target network ושני מצבי Adam זה עדיין עשרות MB בודדות, ו-minibatch של 32 תמונות `84×84×4` הוא זניח.
- ה-replay buffer יושב בדרך כלל בזיכרון של ה-CPU (RAM), לא ב-GPU. ב-SB3 הוא מערך numpy [20], ו-CleanRL בונים buffer באותו ממשק, עם `optimize_memory_usage=True` [15].

### 5.2 זיכרון RAM: כאן צריך להיזהר

במחשב יש בערך 32 GB של RAM (נמדד במכונה). **חישוב** (uint8, פריים של `84×84` הוא 7,056 בייט):

| שיטת אחסון | בייטים למעבר | 1M מעברים | 100K מעברים |
|---|---|---|---|
| נאיבי: `s` ו-`s'`, כל אחד 4 פריימים | 56,448 | ~56 GB (לא נכנס) | ~5.6 GB |
| `s` בלבד, ו-`s'` מהמעבר הבא (`optimize_memory_usage` של SB3, "almost a factor two" [20]) | 28,224 | ~28 GB (גבולי מאוד) | ~2.8 GB |
| כל פריים נשמר פעם אחת, והערימה נבנית בזמן הדגימה (כמו Dopamine [12]) | 7,056 | ~7 GB | ~0.7 GB |

- ב-RL Zoo מגיעים ל-20.6 ב-Pong עם buffer של 100K בלבד [16][17].
- **לא ודאי:** בכרטיס של המודל המאומן (model card) מופיע buffer של 10K [18], ובקובץ ה-config הנוכחי מופיע 100K [17]. ייתכן שה-config השתנה מאז שהמודל אומן.
- **המשמעות:** גודל ה-buffer הוא החלטה אמיתית לטיקט ה-hyperparameters, והוא תלוי בשיטת האחסון.

### 5.3 זמן שעון: הערכה גסה בלבד

נקודות עוגן מהמקורות (כולן על ALE, עם אמולטור):

- **2013:** ‏epoch של 50,000 עדכוני minibatch לקח "roughly 30 minutes" [2]. **חישוב:** עם עדכון כל 4 צעדים, ‏10M צעדים הם 2.5M עדכונים, כלומר כ-25 שעות על חומרה של 2013.
- ‏**Dopamine:** ‏DQN על Pong רץ ב-"around 800 fps" על Tesla P100 [8]. **לא ודאי** אם מדובר בפריימים של המשחק (כ-200 צעדים בשנייה, ו-10M צעדים בכ-14 שעות) או בצעדי Agent (כ-3.5 שעות).
- ‏**Revisiting Rainbow:** משחק ALE לוקח "roughly 5 days to fully train" על חומרה כמו P100 [13]. **חישוב:** בהנחה שאימון מלא הוא 200M פריימים (ההנחה שלי), זה כ-116 צעדים בשנייה, כלומר כ-24 שעות ל-10M צעדים.
- ‏**Rainbow:** ‏7M פריימים לקחו פחות מ-10 שעות [6].

השוואה לכרטיס שלנו:

- ל-P2200 יש 3.8 TFLOPS ב-FP32 [23], ול-P100 PCIe יש 9.3 [24]. ה-P2200 חלש בערך פי 2.5.
- **הערכה שלי:** ברשת כזו קטנה ועם batch של 32, ה-GPU כנראה לא יהיה צוואר הבקבוק. סביר יותר שזה יהיה Python, הסביבה, והעברת הנתונים ל-GPU. ‏SB3 מציינים שאלגוריתמים עם replay buffer, כמו DQN, הם "usually slower to train (regarding wall clock time)" [21].
- **השורה התחתונה (לא ודאית):** כדאי לתכנן כמה שעות עד יום-יומיים לריצה של עד 10M צעדים. צריך למדוד צעדים לשנייה כבר בשבוע הראשון. זה קשור לסעיף "יעד מהירות לסביבה" במפה.

### 5.4 הערה לטיקט הכלים (#2)

‏PyTorch הודיעו שגרסה 2.14 היא האחרונה עם wheels מוכנים ל-Pascal (‏sm_60), דרך CUDA 12.6. מגרסה 2.15 צריך לבנות מהמקור [25]. הכרטיס שלנו הוא compute capability 6.1 (נמדד במכונה). **לא בדקתי** אם יש wheel של CUDA 12.6 ל-Python 3.14. זו שאלה ל-#2.

## 6. איך Pong שלנו שונה מ-ALE Pong

כל המספרים למעלה הם מ-ALE Pong. ההבדלים הצפויים (רובם ניתוח שלי, ומקורות איפה שיש):

- **אין אמולטור:** אפשר לצייר ישר לזיכרון ברזולוציה קטנה, ולכן צעד של הסביבה זול יותר (**הערכה**).
- **אין flicker:** הסיבה ל-max על שני פריימים הייתה flicker של Atari [1]. אם המשחק שלנו מצייר כל אובייקט בכל פריים, השלב הזה מיותר.
- ‏**Reward clipping כבר קורה מעצמו:** הסיבה לחיתוך הייתה סקאלות ניקוד שונות בין משחקים [1]. אצלנו ה-Reward הוא `+1` או `-1` על כל Point, כך שהוא כבר בטווח.
- **אין חיים:** אין צורך לסיים episode באיבוד חיים. גם Machado ממליצים להשתמש רק בסיום המשחק [7].
- **דטרמיניזם:** ‏ALE הוא "fully deterministic", ו-agent יכול פשוט לשנן רצף פעולות [7]. ב-Nature הוסיפו no-op starts ו-ε=0.05 כדי להקטין overfitting בהערכה [1]. אצלנו ה-Evaluation היא בלי אקראיות. אם גם המשחק וגם ה-Opponent דטרמיניסטיים, כל 100 ה-Games עלולים להיות זהים. צריך מקור אקראיות בסביבה, למשל זווית ההגשה. זו שאלה לטיקטים #4 ו-#7.
- **ה-Opponent קובע אם היעד אפשרי בכלל:** ב-ALE ה-DQN מגיע ל-18.9 עד 21 [1][6], כלומר ה-Opponent של Atari ניתן לניצחון. אם ה-Opponent שלנו לא מחטיא אף פעם, היעד בלתי אפשרי. ואם הוא חלש מאוד, היעד קל מדי. שאלה לטיקט #7.
- **צפיפות ה-Reward:** כמה החלטות עוברות בין Point ל-Point תלוי בתדירות ההחלטה (#8) ובמהירות הכדור (#4). **הערכה:** פחות החלטות לכל Point מקלות על הקצאת האשראי. ‏n-step עוזר כשיש הרבה החלטות.
- **ערימת פריימים:** בלי כמה פריימים, ה-Agent לא יכול לדעת לאן הכדור נע [1][2]. שאלה לטיקט #6.

## 7. טעויות נפוצות שגורמות ל-DQN לא ללמוד

1. ‏**Bootstrap דרך סוף ה-Game.** במצב סופי ה-target הוא רק `r` [1]. מצד שני, אם קוטעים Game באמצע בגלל מגבלת זמן, זה לא סוף אמיתי, וצריך לסמן אותו כ-truncated [21].
2. **פירוש שגוי של error clipping.** חיתוך ה-loss עצמו, במקום Huber, מאפס את הגרדיאנט במעברים עם השגיאה הגדולה [22].
3. **באגים בעיבוד התמונה.** ‏OpenAI מצאו באג בהמרה לגווני אפור, שבגללו הדגים נעלמו מהמסך ב-Seaquest. ההמלצה שלהם היא לראות את העולם כמו שה-Agent רואה אותו [22].
4. ‏**Observation בלי ערימת פריימים.** אז אי אפשר לדעת לאן הכדור נע [1][2].
5. **בלי replay או בלי target network.** ראו את הירידות הגדולות ב-ablation של Nature [1].
6. **לוח זמנים שגוי ל-ε.** ‏OpenAI כותבים שלירידת ה-ε הייתה השפעה גדולה [22]. ‏Dopamine כותבים ש-1% פעולות אקראיות עדיף בהרבה על 10% [9].
7. **בחירת optimizer.** עם RMSProp ו-Huber, ה-DQN של Dopamine לא הגיע ל-18 ב-50M צעדים. עם Adam ו-MSE הוא הגיע לשם בכ-10M צעדים [10][13].
8. **פיקסלים לא מנורמלים.** ‏CleanRL מחלקים את הקלט ב-255 [15]. ‏SB3 ממליצים "always normalize your observation space if you can" [21].
9. **בלי השוואה ל-Agent אקראי.** "always verify your agent outperforms a random one" [22]. ב-ALE Pong, ‏Agent אקראי מקבל `-20.7` [1].
10. **באגים שקטים של shapes ו-broadcasting.** "a broadcast mistake will fail silently" [21]. דוגמה נפוצה (**הערכה**): צורה של `(B,1)` מול `(B,)` שהופכת בשקט ל-`(B,B)` בחישוב ה-loss.
11. **מסקנות מ-seed אחד.** "results in RL may vary from one run to another" [21]. ראו את הפיזור בין 13.0 ל-19.0 ב-Dopamine [10].
12. **פאניקה מעקומה רועשת.** הציון הממוצע רועש מאוד. ממוצע ה-Q המקסימלי על סט קבוע של מצבים עולה בצורה חלקה יותר, ולכן הוא מדד מעקב טוב [2]. זה רלוונטי לטיקט #10.
13. ‏**Replay buffer שלא נכנס ל-RAM.** ראו את סעיף 5.2.
14. **לכוונן hyperparameters לפני שמתקנים באגים.** ‏OpenAI ממליצים לתקן באגים קודם [22]. ‏SB3 ממליצים להראות "signs of life" על בעיות צעצוע, ולהתקדם לסביבות קשות יותר בהדרגה [21]. זה רלוונטי לטיקט #9.

## 8. המלצה

**לפתוח ב-DQN של Nature 2015, עם שני שינויים קלים: Adam ו-Double DQN.**

מה זה כולל:

- **מה-Nature:** ‏replay buffer אחיד, ‏target network עם העתקה מלאה כל C צעדים, ‏Observation של `84×84` בגווני אפור עם 4 פריימים מוערמים, הרשת 32/64/64/512, ו-ε-greedy שיורד לינארית [1].
- ‏**Adam במקום RMSProp:** כל מימושי הייחוס המודרניים שמגיעים ל-20 ב-Pong משתמשים ב-Adam [14][16][11]. ב-Dopamine זה ההבדל בין "לא מגיע ל-18" לבין "מגיע ב-10M צעדים" [10].
- ‏**Target של Double DQN:** שורה אחת, בלי פרמטרים נוספים [3]. ב-Pong הוא שיפר את הציון מ-18.9 ל-21.0 [3]. כדאי להשאיר אותו כמתג, כדי שאפשר יהיה לראות את ההשפעה שלו ולהבין overestimation.
- ‏**Loss:** גם Huber [19] וגם MSE [15] עובדים ב-Pong עם Adam. הבחירה נשארת להחלטת ה-hyperparameters.

למה לא להתחיל עם Rainbow:

- ב-Pong כל הווריאנטים מגיעים לבערך 19.5 עד 21 [4][6]. היעד שלנו הוא בערך הרמה של DQN המקורי [1], כך ש-Rainbow לא נדרש כדי להגיע אליו.
- ‏Rainbow מקצר את האימון בערך פי 2.5 בצעדים [10]. אבל הוא מוסיף sum-tree, משקלי IS, התפלגויות (C51) ו-noisy layers [5][6]. בפרויקט לימודי כל רכיב כזה הוא עוד מקום לבאג שקט.
- ‏DQN פשוט הוא גם מה שהמפה ביקשה: "ברוח ה-DQN של DeepMind".

מסלול שדרוג, אם האימון איטי מדי לתקציב שייקבע:

1. ‏**n-step returns** (למשל n=3). אחד משני הרכיבים הכי חשובים ב-Rainbow [6], ופשוט יחסית לממש (**הערכה**).
2. ‏**Prioritized Replay.** בערך פי 2 מהר יותר בחציון [5], אבל דורש sum-tree.
3. ‏**Dueling.** פשוט לממש, אבל עם 3 פעולות כנראה יתרון קטן [4][6].
4. **לא לגעת** ב-Distributional וב-Noisy Nets, אלא אם יש סיבה ספציפית.

מה נשאר פתוח ולא נקבע כאן:

- ערכי ה-hyperparameters: קצב למידה, גודל buffer, לוח הזמנים של ε, ותדירות עדכון ה-target. הם נשארים ב-Not yet specified של המפה. טבלת 2015 בסעיף 2.5 היא רק נקודת ייחוס.

## 9. מה לא ודאי

- **מספרי הצעדים והזמן הם מ-ALE Pong.** ב-Pong שלנו הם יכולים להיות קטנים או גדולים יותר, לפי ה-Opponent, הפיזיקה ותדירות ההחלטה (#4, ‏#7, ‏#8).
- **נתוני Dopamine** הם ציוני אימון עם ε=0.01 ו-sticky actions [11]. הם לא ציון של Evaluation עם ε=0 על Games מלאים.
- **ההגדרות של "frame" לא אחידות** בין המקורות [7]. בפרט, לא ברור אם ה-800 fps של Dopamine [8] הם פריימים של המשחק או צעדי Agent. גם תדירות ה-target של Nature לא חד-משמעית: ‏"parameter updates" [1] מול "agent steps" [9].
- **אין מקור ראשוני שנותן קיצור זמן ספציפי ל-Pong** עבור Double, ‏Dueling או PER בנפרד. ההאצות שמדווחות הן חציון על כל המשחקים [5][6]. ההשוואה ל-Pong היחידה שמצאתי היא עקומות Dopamine [10].
- **זמן השעון על P2200 הוא הערכה גסה בלבד** (סעיף 5.3). צריך למדוד.
- **מספר ה-seeds** בתוצאה של CleanRL `dqn_atari.py` לא מצוין [14]. גודל ה-buffer של מודל ה-Pong ב-RL Zoo לא עקבי בין ה-model card ל-config [17][18].
- ‏**Python 3.14 ו-Pascal:** לא נבדק אם יש wheel מתאים של PyTorch [25].

## מקורות

1. Mnih et al., 2015. "Human-level control through deep reinforcement learning", Nature 518. ‏PDF: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf ‏(Methods; Extended Data Tables 1–3)
2. Mnih et al., 2013. "Playing Atari with Deep Reinforcement Learning". https://arxiv.org/abs/1312.5602
3. van Hasselt, Guez, Silver, 2015. "Deep Reinforcement Learning with Double Q-learning". https://arxiv.org/abs/1509.06461
4. Wang et al., 2016. "Dueling Network Architectures for Deep Reinforcement Learning". https://arxiv.org/abs/1511.06581
5. Schaul et al., 2016. "Prioritized Experience Replay". https://arxiv.org/abs/1511.05952
6. Hessel et al., 2018. "Rainbow: Combining Improvements in Deep Reinforcement Learning". https://arxiv.org/abs/1710.02298
7. Machado et al., 2018. "Revisiting the Arcade Learning Environment". https://arxiv.org/abs/1709.06009
8. Castro et al., 2018. "Dopamine: A Research Framework for Deep Reinforcement Learning". https://arxiv.org/abs/1812.06110
9. Dopamine, Atari baselines README. https://github.com/google/dopamine/blob/master/baselines/atari/README.md
10. Dopamine, Pong baseline data. https://github.com/google/dopamine/blob/master/baselines/atari/data/pong.json ‏(החישובים בסעיף 4.2 הם שלי)
11. Dopamine configs: https://github.com/google/dopamine/blob/master/dopamine/jax/agents/dqn/configs/dqn.gin , https://github.com/google/dopamine/blob/master/dopamine/tf/agents/dqn/configs/dqn.gin , https://github.com/google/dopamine/blob/master/dopamine/jax/agents/rainbow/configs/rainbow.gin
12. Dopamine, `circular_replay_buffer.py`. https://github.com/google/dopamine/blob/master/dopamine/tf/replay_memory/circular_replay_buffer.py
13. Obando-Ceron & Castro, 2021. "Revisiting Rainbow". https://arxiv.org/abs/2011.14826
14. CleanRL DQN docs. https://docs.cleanrl.dev/rl-algorithms/dqn/
15. CleanRL `dqn_atari.py`. https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
16. RL Baselines3 Zoo, `benchmark.md`. https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
17. RL Baselines3 Zoo, `hyperparams/dqn.yml`. https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
18. Model card `sb3/dqn-PongNoFrameskip-v4`. https://huggingface.co/sb3/dqn-PongNoFrameskip-v4
19. Stable-Baselines3 DQN docs ו-source. https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html , https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py
20. Stable-Baselines3, `buffers.py`. https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
21. Stable-Baselines3, "Reinforcement Learning Tips and Tricks". https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
22. OpenAI, 2017. "OpenAI Baselines: DQN". https://openai.com/index/openai-baselines-dqn/
23. PNY, NVIDIA Quadro P2200 spec. https://www.pny.com/nvidia-quadro-p2200
24. NVIDIA Tesla P100 PCIe datasheet. https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf
25. PyTorch dev-discuss, "Notice: CUDA 12.6 wheels will no longer be published from PyTorch 2.15". https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432
