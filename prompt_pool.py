DATASET_PROMPTS = {
    "test": "Question: {input_data}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
    "GSM8K": "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"Answer:\". Do not add anything other than the integer answer after \"Answer:\".\n\nQuestion:\n{input_data}\n",
    "mgsm": "Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of \"Answer:\". Do not add anything other than the integer answer after \"Answer:\".\n\nQuestion:\n{input_data}\n",
    "math": "Question: {input_data}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
    "math500": "Question: {input_data}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n",
    "mmmlu": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nQuestion:\n{input_data}\n",
    "belebele": "Answer the following multiple choice reading-comprehension question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Please fully understand the passage and give explanations step by step before answering.\n\n{input_data}\n",
    "commonsenseqa": "Answer the following multiple choice common-sense reasoning question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step and output the reasoning process before answering.\n\n{input_data}",
    "theoremqa": '''
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Please read a math problem, and then think step by step to derive the answer. The answer is decided by Answer Type.
If the Answer type in [bool], the answer needs to be True or False.
Else if the Answer type in [integer, float] , The answer needs to be in numerical form.
Else if the Answer type in [list of integer, list of float] , the answer needs to be a list of number like [2, 3, 4].
Else if the Answer type in [option], the answer needs to be an option like (a), (b), (c), (d).
You need to output the answer in your final sentence like 'Therefore, the answer is ...'.

### Question: 
''' + "{input_data}\n\n### Answer_type: {answer_type}\n\n### Response:"
}



QUESTIONS = {
    "en": "Question:",
    "es": "Pregunta:",
    "fr": "Question:",
    "de": "Frage:",
    "ru": "Задача:",
    "zh": "问题:",
    "ja": "問題:",
    "th": "โจทย์:",
    "sw": "Swali:",
    "bn": "প্রশ্ন:",
    "te": "ప్రశ్న:"
}

ANSWERS = {
    "en": "Step-by-Step Answer:",
    "es": "Respuesta paso a paso:",
    "fr": "Réponse étape par étape :",
    "de": "Schritt-für-Schritt-Antwort:",
    "ru": "Пошаговоерешение:",
    "zh": "逐步解答:",
    "ja": "ステップごとの答え:",
    "th": "คำตอบทีละขั้นตอน:",
    "sw": "Jibu la Hatua kwa Hatua:",
    "bn": "ধাপে ধাপে উত্তর:",
    "te": "దశలవారీగా సమాధానం:"
}


INSTRUCTIONS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.""",
    "de": """Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.""",
    "es": """Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".""",
    "fr": """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".""",
    "ja": """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。""",
    "ru": """Решите эту математическую задачу. Объясните шаги рассуждения перед тем, как дать окончательный ответ в последней строке сам по себе в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа после "Ответ:".""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.""",
    "th": """แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:""""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。""",
}

ANSWER_PREFIX = {
    "en": "Answer",
    "bn": "উত্তর",
    "de": "Antwort",
    "es": "Respuesta",
    "fr": "Réponse",
    "ja": "答え",
    "ru": "Ответ",
    "sw": "Jibu",
    "te": "సమాధానం",
    "th": "คำตอบ",
    "zh": "答案",
}


