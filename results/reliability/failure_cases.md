# Validation diagnostic cases

No intermediate reasoning was manually certified. These are answer-level observations, not additional training labels.

## both model and majority wrong (validation row 32)

Question: Yun had 20 paperclips initially, but then lost 12. Marion has 1/4 more than what Yun currently has, plus 7. How many paperclips does Marion have?

Selected candidate: 3; parsed answer: 17; reference: 9; confidence: 0.9547.

Observed answer/selection mismatch only; intermediate reasoning has not been manually verified. No process-supervision label was created.

```text
We are given the following:

- Yun had **20 paperclips** initially.
- Then lost **12**, so **Yun currently has** $ 20 - 12 = 8 $ paperclips.

- Marion has **$ \frac{1}{4} $ more than what Yun currently has, plus 7**.

So, we calculate:

$$
\text{Marion's paperclips} = 8 + \left( \frac{1}{4} \times 8 \right) + 7
$$

$$
= 8 + 2 + 7 = 17
$$

### Final Answer:
$$
\boxed{17}
$$
```

## model wrong, majority correct (validation row 29)

Question: Megan bought 2 dozen eggs. As she was walking to her car, she dropped a tray of eggs. 3 eggs broke, and twice as many cracked. What is the difference between the eggs that are still in perfect condition and those that are cracked?

Selected candidate: 2; parsed answer: 6; reference: 9; confidence: 0.4467.

Observed answer/selection mismatch only; intermediate reasoning has not been manually verified. No process-supervision label was created.

```text
Megan bought 2 dozen eggs, which is **24 eggs**. She dropped a tray of eggs, and **3 eggs broke**, and **twice as many cracked**, meaning **6 eggs cracked**.

- Perfect condition: $ 24 - 3 - 6 = 15 $ eggs.
- Cracked: $ 3 + 6 = 9 $ eggs.

The difference between the eggs that are still in perfect condition and those that are cracked is:

$$
15 - 9 = \boxed{6}
$$

#### <numeric answer> 6
```

## model correct, majority wrong (validation row 3)

Question: Jay went to watch a singer in a one hour 20 minutes concert.  If there was a 10-minute intermission, and all the songs were 5 minutes except for one song that lasted 10 minutes, how many songs did she sing?

Selected candidate: 2; parsed answer: 13; reference: 13; confidence: 0.5620.

Observed answer/selection mismatch only; intermediate reasoning has not been manually verified. No process-supervision label was created.

```text
Jay attended a concert that started at 9:00 AM and ended at 9:40 AM. This is a **1-hour 20-minute** concert, which means the total duration is:

$$
60 + 20 = 80 \text{ minutes}
$$

The concert includes an **intermission** of 10 minutes, so the actual time she sang is:

$$
80 - 10 = 70 \text{ minutes}
$$

Each song is **5 minutes** except for **one song that lasted 10 minutes**. Let’s denote the number of songs she sang as $ x $.

- The time taken for the **other $ x - 1 $** songs is $ 5(x - 1) $ minutes.
- The time for the **10-minute song** is 10 minutes.

So the total time is:

$$
5(x - 1) + 10 = 70
$$

Solving:

$$
5x - 5 + 10 = 70 \\
5x + 5 = 70 \\
5x = 65 \\
x = 13
$$

### Final Answer:

$$
\boxed{13}
$$
```

## additional high-confidence error (validation row 586)

Question: On a particular day, Rose's teacher read the register and realized there were twice as many girls as boys present on that day. The class has 250 students, and all the 140 girls were present. If all the absent students were boys, how many boys were absent that day?

Selected candidate: 2; parsed answer: 180; reference: 40; confidence: 0.9544.

Observed answer/selection mismatch only; intermediate reasoning has not been manually verified. No process-supervision label was created.

```text
We are given the following:

- Total number of students = 250  
- Number of girls present = 140  
- Number of girls is **twice** as many as boys present.  
- All the **absent students** are boys.  
- We are to find how many **boys were absent**.

---

### Step 1: Let the number of boys present be $ x $.

Since there are **twice as many girls as boys present**, we have:

$$
\text{Number of girls present} = 2x
$$

But we are told the number of girls present is **140**. So:

$$
2x = 140 \Rightarrow x = 70
$$

So, there are **70 boys present**.

---

### Step 2: Total number of students is 250.

$$
\text{Total students} = \text{Present students} + \text{Absent students}
$$

$$
250 = 70 + \text{Absent students}
$$

$$
\text{Absent students} = 250 - 70 = 180
$$

---

### ✅ Final Answer:

$$
\boxed{180}
$$
```