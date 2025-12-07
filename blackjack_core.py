import base64
import io
import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# ================== Config ==================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DECKS_IN_SHOE = 6
HIT_SOFT_17 = True          # dealer hits on soft 17
DOUBLE_AFTER_SPLIT = True
SURRENDER_ALLOWED = True

MODEL = "gpt-4o-mini"

# ================== Card + Count Data ==================

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["♠", "♥", "♦", "♣"]

RANK_TO_VALUE = {
    "A": 11, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10
}

HILO = {
    "2": +1, "3": +1, "4": +1, "5": +1, "6": +1,
    "7": 0, "8": 0, "9": 0,
    "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1
}

ACTION_CODE_TO_TEXT = {
    "H": "Hit",
    "S": "Stand",
    "D": "Double",
    "P": "Split",
    "R": "Surrender",
}


@dataclass
class VisionHand:
    cards: List[str]

    def total(self) -> Tuple[int, bool]:
        if not self.cards:
            return 0, False

        values = [RANK_TO_VALUE[c[:-1]] for c in self.cards]
        total = sum(values)
        aces = sum(1 for c in self.cards if c.startswith("A"))
        soft = False

        while total > 21 and aces:
            total -= 10
            aces -= 1

        if any(c.startswith("A") for c in self.cards) and total <= 21:
            raw_sum = sum(RANK_TO_VALUE[c[:-1]] for c in self.cards)
            if raw_sum != total:
                soft = True

        return total, soft


@dataclass
class ShoeState:
    decks: float = DECKS_IN_SHOE
    running_count: int = 0
    cards_seen: int = 0

    def true_count(self) -> float:
        decks_remaining = max(1e-9, self.decks_remaining())
        return self.running_count / decks_remaining

    def decks_remaining(self) -> float:
        if self.cards_seen == 0:
            return self.decks
        est = max(0.5, self.decks * (1.0 - (self.cards_seen / (52.0 * self.decks))))
        return est

    def see_cards(self, cards: List[str]):
        for c in cards:
            if not c:
                continue
            rank = c[:-1]
            if rank in HILO:
                self.running_count += HILO[rank]
            self.cards_seen += 1


@dataclass
class VisionResult:
    dealer_up: Optional[str]
    dealer_down_known: Optional[str]
    player: List[str]
    extra_player_hands: List[List[str]] = field(default_factory=list)
    dealer_all_cards: Optional[List[str]] = None


# ================== Helpers ==================

def dealer_up_value(card: str) -> int:
    return RANK_TO_VALUE[card[:-1]]


def is_pair(cards: List[str]) -> bool:
    return len(cards) == 2 and cards[0][:-1] == cards[1][:-1]


def basic_strategy_action(player_cards: List[str], dealer_up: str) -> str:
    up = dealer_up_value(dealer_up)
    total, soft = VisionHand(player_cards).total()

    if is_pair(player_cards):
        r = player_cards[0][:-1]
        if r == "A" or r == "8": return "P"
        if r == "10": return "S"
        if r == "9": return "P" if up in [2, 3, 4, 5, 6, 8, 9] else "S"
        if r == "7": return "P" if up in [2, 3, 4, 5, 6, 7] else "H"
        if r == "6": return "P" if up in [2, 3, 4, 5, 6] else "H"
        if r == "5": return "D" if up in [2, 3, 4, 5, 6, 7, 8, 9] else "H"
        if r == "4": return "P" if up in [5, 6] else "H"
        if r in ["3", "2"]: return "P" if up in [2, 3, 4, 5, 6, 7] else "H"

    if soft:
        if total >= 19: return "S"
        if total == 18:
            return "S" if up in [2, 7, 8] else ("D" if up in [3, 4, 5, 6] else "H")
        if total == 17: return "D" if up in [3, 4, 5, 6] else "H"
        if total in [15, 16]: return "D" if up in [4, 5, 6] else "H"
        if total in [13, 14]: return "D" if up in [5, 6] else "H"
        return "H"

    if total >= 17: return "S"
    if total == 16:
        return "S" if up in [2, 3, 4, 5, 6] else (
            "R" if SURRENDER_ALLOWED and up in [9, 10, 11] else "H"
        )
    if total == 15:
        return "S" if up in [2, 3, 4, 5, 6] else (
            "R" if SURRENDER_ALLOWED and up == 10 else "H"
        )
    if total in [13, 14]: return "S" if up in [2, 3, 4, 5, 6] else "H"
    if total == 12: return "S" if up in [4, 5, 6] else "H"
    if total == 11: return "D"
    if total == 10: return "D" if up in [2, 3, 4, 5, 6, 7, 8, 9] else "H"
    if total == 9:  return "D" if up in [3, 4, 5, 6] else "H"
    return "H"


def apply_count_deviations(action: str, tc: float,
                           player_cards: List[str],
                           dealer_up: str) -> str:
    up = dealer_up_value(dealer_up)
    total, soft = VisionHand(player_cards).total()

    if not soft and total == 16 and up == 10 and tc >= 0:
        return "S"
    if not soft and total == 15 and up == 10 and tc >= 4:
        return "S"
    if not soft and total == 12 and up == 3 and tc >= 2:
        return "S"
    if not soft and total == 12 and up == 2 and tc >= 3:
        return "S"
    if not soft and total == 10 and up == 10 and tc >= 4:
        return "D"

    return action


def betting_units_from_tc(tc: float) -> int:
    if tc <= 0: return 1
    if tc < 2:  return 2
    if tc < 3:  return 4
    if tc < 5:  return 6
    if tc < 7:  return 8
    return 12


def estimate_win_probability(tc: float) -> float:
    base = 0.43
    adj = 0.015 * tc
    p = base + adj
    return max(0.25, min(0.70, p))


def format_total_string(cards: List[str]) -> str:
    if not cards:
        return "0"
    total, soft = VisionHand(cards).total()
    if soft and total > 11:
        lower = total - 10
        return f"{lower}/{total}"
    return f"{total}"


# ================== OpenAI / Vision ==================

def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def create_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
    return OpenAI(api_key=OPENAI_API_KEY)


def call_vision_for_cards(client: OpenAI, img: Image.Image) -> VisionResult:
    img_b64 = encode_image(img)

    system = (
        "You are a blackjack card reader. You ONLY return JSON. "
        "Detect visible playing cards for dealer and player from the screenshot. "
        "Card format: 'RANKSUIT' with RANK in {A,2..10,J,Q,K} and SUIT in {♠,♥,♦,♣}. "
        "If dealer downcard is face-down or uncertain, set dealer_down_known to null. "
        "If the image is split into two regions, assume the LEFT area is dealer and the RIGHT area is player. "
        "Return keys: dealer_up, dealer_down_known, player, extra_player_hands, dealer_all_cards."
    )

    user_content = [
        {
            "type": "text",
            "text": (
                "Return JSON ONLY with keys: dealer_up, dealer_down_known, "
                "player, extra_player_hands, dealer_all_cards.\n"
                "Example: {"
                "\"dealer_up\":\"10♣\","
                "\"dealer_down_known\":null,"
                "\"player\":[\"9♦\",\"A♠\"],"
                "\"extra_player_hands\":[],"
                "\"dealer_all_cards\":[\"10♣\"]"
                "}"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        },
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    content = resp.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].lstrip()

    data = json.loads(content)

    dealer_up = data.get("dealer_up")
    dealer_down_known = data.get("dealer_down_known")
    player = data.get("player") or []
    extra_hands = data.get("extra_player_hands") or []
    dealer_all = data.get("dealer_all_cards")

    return VisionResult(
        dealer_up=dealer_up,
        dealer_down_known=dealer_down_known,
        player=player,
        extra_player_hands=extra_hands,
        dealer_all_cards=dealer_all,
    )


def explain_recommendation(client: OpenAI, context: Dict) -> str:
    system = (
        "You are an expert blackjack coach. "
        "Explain recommendations briefly in casual, clear language. "
        "Use at most 3 short sentences. Don't repeat the raw JSON."
    )

    user_text = (
        "Here is the game state in JSON:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Explain why the recommended action is best, mentioning the hand totals "
        "and how the true count and approximate win chance influence it."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ================== Core Recommend Helper ==================

def _recommend_from_vision_result(vision: VisionResult,
                                  shoe: ShoeState,
                                  client: Optional[OpenAI],
                                  include_explanation: bool,
                                  update_shoe: bool) -> Dict:
    visible_cards: List[str] = []
    if vision.dealer_up:
        visible_cards.append(vision.dealer_up)
    if vision.dealer_all_cards:
        for c in vision.dealer_all_cards[1:]:
            visible_cards.append(c)
    for c in vision.player:
        visible_cards.append(c)
    for hand in vision.extra_player_hands:
        visible_cards.extend(hand)

    if update_shoe:
        shoe.see_cards(visible_cards)

    tc = round(shoe.true_count(), 2)

    if vision.dealer_all_cards:
        dealer_cards = vision.dealer_all_cards
    else:
        dealer_cards = []
        if vision.dealer_up:
            dealer_cards.append(vision.dealer_up)
        if vision.dealer_down_known:
            dealer_cards.append(vision.dealer_down_known)

    player_cards = vision.player or []

    dealer_total_str = format_total_string(dealer_cards)
    player_total_str = format_total_string(player_cards)

    win_prob = estimate_win_probability(tc)

    def build_base_dict(action_code: Optional[str],
                        bet_units: int,
                        insurance: bool,
                        explanation: Optional[str]) -> Dict:
        return {
            "vision_raw": {
                "dealer_up": vision.dealer_up,
                "dealer_down_known": vision.dealer_down_known,
                "player": vision.player,
                "extra_player_hands": vision.extra_player_hands,
                "dealer_all_cards": vision.dealer_all_cards,
                "dealer_total": dealer_total_str,
                "player_total": player_total_str,
            },
            "counting": {
                "running_count": shoe.running_count,
                "true_count": tc,
                "decks_remaining_est": round(shoe.decks_remaining(), 2),
                "cards_seen": shoe.cards_seen,
            },
            "recommendation": {
                "bet_units": bet_units,
                "action": action_code,
                "insurance": insurance,
            },
            "win_probability": win_prob,
            "explanation": explanation,
        }

    if not player_cards or not vision.dealer_up:
        base = build_base_dict(None, betting_units_from_tc(tc), False, None)
        if include_explanation and client is not None:
            ctx = {
                "dealer_cards": dealer_cards,
                "player_cards": player_cards,
                "dealer_total": dealer_total_str,
                "player_total": player_total_str,
                "action_code": None,
                "action_text": None,
                "true_count": tc,
                "running_count": shoe.running_count,
                "win_probability": win_prob,
            }
            base["explanation"] = explain_recommendation(client, ctx)
        return base

    base_action = basic_strategy_action(player_cards, vision.dealer_up)
    final_action = apply_count_deviations(base_action, tc, player_cards, vision.dealer_up)
    insurance = (dealer_up_value(vision.dealer_up) == 11 and tc >= 3)
    units = betting_units_from_tc(tc)

    explanation = None
    result = build_base_dict(final_action, units, insurance, None)

    if include_explanation and client is not None:
        ctx = {
            "dealer_cards": dealer_cards,
            "player_cards": player_cards,
            "dealer_total": dealer_total_str,
            "player_total": player_total_str,
            "action_code": final_action,
            "action_text": ACTION_CODE_TO_TEXT.get(final_action),
            "true_count": tc,
            "running_count": shoe.running_count,
            "win_probability": win_prob,
        }
        explanation = explain_recommendation(client, ctx)
        result["explanation"] = explanation

    return result


# ================== Public API ==================

def recommend_from_image(img: Image.Image,
                         shoe: ShoeState,
                         client: OpenAI,
                         include_explanation: bool = True) -> Dict:
    vision = call_vision_for_cards(client, img)
    return _recommend_from_vision_result(
        vision,
        shoe,
        client,
        include_explanation=include_explanation,
        update_shoe=True,
    )


def recommend_from_cards(vision_raw: Dict,
                         shoe: ShoeState,
                         client: Optional[OpenAI],
                         include_explanation: bool = True,
                         update_shoe: bool = False) -> Dict:
    """
    Same output as recommend_from_image, but using a pre-edited card set
    (dealer_up, dealer_down_known, player, extra_player_hands, dealer_all_cards).

    Used by the GUI when you manually tweak cards or add new ones.
    update_shoe=False by default so we don't double-count cards.
    """
    vision = VisionResult(
        dealer_up=vision_raw.get("dealer_up"),
        dealer_down_known=vision_raw.get("dealer_down_known"),
        player=vision_raw.get("player") or [],
        extra_player_hands=vision_raw.get("extra_player_hands") or [],
        dealer_all_cards=vision_raw.get("dealer_all_cards"),
    )

    return _recommend_from_vision_result(
        vision,
        shoe,
        client,
        include_explanation=include_explanation,
        update_shoe=update_shoe,
    )
