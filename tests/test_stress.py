from ukrainian_tts.stress import sentence_to_stress, stress_with_model


def test_stress_table():
    examples = [
        #("Бабин біб розцвів у дощ — Буде бабі біб у борщ.\n\nБоронила", "Б+абин б+іб розцв+ів +у д+ощ — Б+уде б+абі б+іб +у б+орщ.\n\nБорон+ила"),
        #("Бабин біб розцвів у дощ — Буде бабі біб у борщ.,,Боронила", "Б+абин б+іб розцв+ів +у д+ощ — Б+уде б+абі б+іб +у б+орщ.,,Борон+ила"),
        ("Бобер на березі з бобренятами бублики пік.","Боб+ер н+а березі з бобрен+ятами б+ублики п+ік."),
        (
            "Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району.", 
            "Кам'ян+ець-Под+ільський - м+істо в Хмельн+ицькій +області Укра+їни, ц+ентр Кам'ян+ець-Под+ільської міськ+ої об'+єднаної територі+альної гром+ади +і Кам'ян+ець-Под+ільського рай+ону."),
        ("Привіт, як тебе звати?", "Прив+іт, +як теб+е зв+ати?"),
        ("АННА - український панк-рок гурт", "+АННА - укра+їнський панк-р+ок г+урт"),
        ("Не тільки в Україні таке може бути.", "Н+е т+ільки в Укра+їні так+е м+оже б+ути."),
        ("Не тільки в +Укра+їні т+аке може бути.", "Н+е т+ільки в +Укра+їні т+аке м+оже б+ути."),
        ("два + два", "дв+а + дв+а"),
        ("Н тльк в крн тк мж бт.", "Н тльк в крн тк мж бт."),

    ]
    for item in examples:
        assert sentence_to_stress(item[0]) == item[1]

    examples = [
        (
            "Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району.", 
            "к+ам'янець-под+ільський - м+істо в хм+ельницькій обл+асті укра+їни, ц+ентр к+ам'янець-под+ільської м+іської об'+єднаної територі+альної гром+ади +і к+ам'янець-под+ільського рай+ону."
        ),
        ("Привіт, як тебе звати?", "прив+іт, +як т+ебе зв+ати?"),
        ("АННА - український панк-рок гурт", "+анна - укра+їнський п+анк-р+ок г+урт"),
        ("Не тільки в Україні таке може бути.", "н+е т+ільки в укра+їні т+аке м+оже б+ути."),
        ("Не тільки в +Укра+їні т+аке може бути.", "н+е т+ільки в ++укра++їні т+ак+е м+оже б+ути."),
        ("два + два", "дв+а + дв+а"),
        ("Н тльк в крн тк мж бт.", "н тльк в крн тк мж бт."),
    ]

    for item in examples:
        assert stress_with_model(item[0]) == item[1]