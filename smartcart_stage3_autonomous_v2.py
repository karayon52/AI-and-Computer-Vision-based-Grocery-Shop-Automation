import io
import time
import wave
from collections import defaultdict, deque
from typing import Dict, Optional
from difflib import get_close_matches

import cv2
import cvzone
import pyaudio
import serial
from PIL import Image
from ultralytics import YOLO
from google import genai
from google.genai import types

# =========================================================
# SMARTCART STAGE 3 - AUTONOMOUS SORTING
# ---------------------------------------------------------
# Flow:
# 1) Take customer orders by manual / image / voice
# 2) For voice: parse -> compare with supported items -> combine
# 3) Assign up to 3 active baskets, extra customers wait
# 4) Run YOLO conveyor detection
# 5) Find which active customer needs detected item next
# 6) Route to that customer's basket through ESP32 or simulation
# 7) Update delivered count automatically
# 8) Complete orders and promote waiting customers automatically
# =========================================================

# =========================
# SETTINGS
# =========================
GEMINI_API_KEY = "ADD_your_gemini_key"

MAX_ACTIVE_CUSTOMERS = 3

# Cameras
SCANNER_CAM_INDEX = 1
SORTING_CAM_INDEX = 0

# YOLO
MODEL_PATH = r"D:\pythonProject\object_detection_practice\obj_det_yolo\grocery\best_april_3_small_model_on_custom.pt"
CLASS_NAMES = ["cola", "soap", "toothpaste"]

FRAME_W = 640
FRAME_H = 480
IMG_SIZE = 480
CONF_THRES = 0.25

# ESP32
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
USE_SERIAL = True   # keep False first, turn True after simulation works

# ROI for conveyor decision
ROI_X1 = 100
ROI_Y1 = 30
ROI_X2 = 500
ROI_Y2 = 400

# Stable decision tuning
MIN_CONF_IN_ZONE = 0.40
MIN_ACCUMULATED_SCORE = 2.2
MIN_OBSERVED_FRAMES = 5
DOMINANCE_RATIO = 1.45
CLEAR_FRAMES_TO_RESET = 6

# Inventory for display / shortage only in this stage
INVENTORY_STOCK = {
    "cola": 10,
    "soap": 10,
    "toothpaste": 10,
}

SUPPORTED_ITEMS = {"cola", "soap", "toothpaste"}

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# NORMALIZATION HELPERS
# =========================
def normalize_item_name(name: str) -> str:
    if not name:
        return ""

    s = name.strip().lower()
    s = " ".join(s.split())
    s = s.replace("-", " ")

    alias_map = {
        "coke": "cola",
        "coca cola": "cola",
        "coca-cola": "cola",
        "cocacola": "cola",
        "coca": "cola",
        "cola": "cola",
        "soft drink": "cola",
        "cold drink": "cola",

        "soap": "soap",
        "saban": "soap",
        "sabun": "soap",
        "shaban": "soap",
        "shavan": "soap",
        "shop": "soap",

        "toothpaste": "toothpaste",
        "tooth paste": "toothpaste",
        "tooth-paste": "toothpaste",
        "paste": "toothpaste",
        "tube": "toothpaste",
        "tubes": "toothpaste",
        "paste tube": "toothpaste",
    }

    return alias_map.get(s, s)


def parse_quantity(text: str) -> int:
    if text is None:
        return 1

    s = str(text).strip().lower()

    word_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "ek": 1, "ekta": 1, "akta": 1,
        "dui": 2, "duita": 2,
        "tin": 3, "tinta": 3,
        "char": 4, "charta": 4,
        "pach": 5, "pachta": 5,
    }

    if s.isdigit():
        return int(s)

    return word_map.get(s, 1)


def parse_csv_text_to_order_dict(csv_text: str) -> Dict[str, int]:
    order = defaultdict(int)

    for raw_line in csv_text.strip().splitlines():
        line = raw_line.strip()
        if not line or "," not in line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue

        item = normalize_item_name(parts[0])
        qty = parse_quantity(parts[1])

        if item and qty > 0:
            order[item] += qty

    return dict(order)


def match_to_supported_item(item_name: str) -> str:
    item_name = normalize_item_name(item_name)

    if item_name in SUPPORTED_ITEMS:
        return item_name

    special_alias = {
        "coco cola": "cola",
        "coco": "cola",
        "sabaan": "soap",
        "sope": "soap",
        "tooth past": "toothpaste",
        "toothpaste": "toothpaste",
        "tooth paste tube": "toothpaste",
    }

    if item_name in special_alias:
        return special_alias[item_name]

    matches = get_close_matches(item_name, list(SUPPORTED_ITEMS), n=1, cutoff=0.55)
    if matches:
        return matches[0]

    return ""


def sanitize_order_to_supported_items(order: Dict[str, int]) -> Dict[str, int]:
    cleaned = defaultdict(int)

    for item, qty in order.items():
        mapped_item = match_to_supported_item(item)
        if mapped_item and qty > 0:
            cleaned[mapped_item] += qty

    return dict(cleaned)


def parse_and_clean_csv_order(csv_text: str) -> Dict[str, int]:
    raw_order = parse_csv_text_to_order_dict(csv_text)
    clean_order = sanitize_order_to_supported_items(raw_order)
    return clean_order


# =========================
# MANUAL ORDER INPUT
# =========================
def manual_order_input() -> Dict[str, int]:
    item_menu = {
        "1": "cola",
        "2": "soap",
        "3": "toothpaste",
    }

    order = defaultdict(int)

    print("\nManual ordering mode")
    print("Press item number, then type quantity.")
    print("1 = cola")
    print("2 = soap")
    print("3 = toothpaste")
    print("0 = finish")

    while True:
        choice = input("Choose item number (0 to finish): ").strip()

        if choice == "0":
            break

        if choice not in item_menu:
            print("Invalid choice. Try again.")
            continue

        item = item_menu[choice]

        try:
            qty = int(input(f"Enter quantity for {item}: ").strip())
        except Exception:
            print("Invalid quantity.")
            continue

        if qty <= 0:
            print("Quantity must be positive.")
            continue

        order[item] += qty
        print(f"Added: {item} x {qty}")

    return dict(order)


# =========================
# IMAGE ORDER INPUT
# =========================
def gemini_extract_order_from_image(image_path: str) -> str:
    img = Image.open(image_path)

    prompt = (
        "Analyze the grocery list in this image. "
        "It may be handwritten Bengali or mixed Bengali-English. "
        "Extract item names and quantities. "
        "Translate item names to English. "
        "Return strictly in CSV format only. "
        "Each line must be: Item, Quantity. "
        "Example:\nSoap, 2\nToothpaste, 1\nCola, 3"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img]
    )
    return response.text.strip()


def auto_scan_paper_and_get_order() -> Dict[str, int]:
    cap = cv2.VideoCapture(SCANNER_CAM_INDEX)
    if not cap.isOpened():
        print("Could not open scanner camera.")
        return {}

    stable_start = 0.0
    is_paper_present = False

    print("\nScanner active. Hold paper steady in front of camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_paper = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 50000:
                current_paper = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if current_paper:
            if not is_paper_present:
                stable_start = time.time()
                is_paper_present = True

            countdown = 2 - (time.time() - stable_start)
            if countdown > 0:
                cv2.putText(
                    frame, f"Steady... {countdown:.1f}s",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
                )
            else:
                capture_path = "autocapture.jpg"
                cv2.imwrite(capture_path, frame)

                cap.release()
                cv2.destroyAllWindows()

                csv_text = gemini_extract_order_from_image(capture_path)
                print("\nGemini extracted order from image:")
                print(csv_text)

                clean_order = parse_and_clean_csv_order(csv_text)

                print("\nCleaned order after supported-item matching:")
                print(clean_order)

                return clean_order
        else:
            is_paper_present = False

        cv2.imshow("SmartCart Auto Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {}


# =========================
# VOICE ORDER INPUT
# =========================
def auto_record_8_seconds(fs: int = 16000) -> bytes:
    p = pyaudio.PyAudio()

    print("\nPreparing recorder...")
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    print("Recording started. Speak your order now.")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=1024
    )

    frames = []
    start_time = time.time()
    duration = 8

    while time.time() - start_time < duration:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    print("Recording ended. Processing audio...")

    stream.stop_stream()
    stream.close()

    sample_width = p.get_sample_size(pyaudio.paInt16)
    p.terminate()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))

    return buffer.getvalue()


def gemini_extract_order_from_audio(audio_bytes: bytes) -> str:
    prompt = (
        "Listen to this grocery order audio. "
        "It may be spoken in Bengali or mixed Bengali-English. "
        "Extract item names and quantities. "
        "Translate item names to English. "
        "Return only CSV lines in this exact format: Item, Quantity. "
        "If quantity is not mentioned, assume 1."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        ]
    )

    return response.text.strip()


def record_voice_and_get_order() -> Dict[str, int]:
    audio_bytes = auto_record_8_seconds()
    csv_text = gemini_extract_order_from_audio(audio_bytes)

    print("\nGemini extracted order from voice:")
    print(csv_text)

    clean_order = parse_and_clean_csv_order(csv_text)

    print("\nMatched and combined order from supported items only:")
    print(clean_order)

    return clean_order


# =========================
# REVIEW / CORRECTION
# =========================
def print_single_order_for_review(order: Dict[str, int]):
    print("\nCurrent extracted order:")
    if not order:
        print("  (empty)")
        return

    for idx, (item, qty) in enumerate(order.items(), start=1):
        support_text = "SUPPORTED" if item in SUPPORTED_ITEMS else "UNSUPPORTED"
        print(f"  {idx}. {item}: {qty} [{support_text}]")


def edit_existing_item(order: Dict[str, int]):
    if not order:
        print("Order is empty.")
        return

    items = list(order.items())
    print_single_order_for_review(order)

    try:
        idx = int(input("Enter item number to edit: ").strip())
    except Exception:
        print("Invalid number.")
        return

    if idx < 1 or idx > len(items):
        print("Invalid item number.")
        return

    old_item, old_qty = items[idx - 1]
    new_item_raw = input(f"Enter new item name for '{old_item}' (or press Enter to keep): ").strip()
    new_qty_raw = input(f"Enter new quantity for '{old_item}' (current {old_qty}) (or press Enter to keep): ").strip()

    new_item = normalize_item_name(new_item_raw) if new_item_raw else old_item
    new_qty = old_qty

    if new_qty_raw:
        try:
            new_qty = int(new_qty_raw)
        except Exception:
            print("Invalid quantity. Keeping old quantity.")

    del order[old_item]

    if new_qty > 0:
        order[new_item] = order.get(new_item, 0) + new_qty
    else:
        print("Quantity <= 0, so item removed.")


def delete_existing_item(order: Dict[str, int]):
    if not order:
        print("Order is empty.")
        return

    items = list(order.items())
    print_single_order_for_review(order)

    try:
        idx = int(input("Enter item number to delete: ").strip())
    except Exception:
        print("Invalid number.")
        return

    if idx < 1 or idx > len(items):
        print("Invalid item number.")
        return

    item_name, _ = items[idx - 1]
    del order[item_name]
    print(f"Removed '{item_name}'.")


def add_new_item(order: Dict[str, int]):
    item_raw = input("Enter new item name: ").strip()
    if not item_raw:
        print("Invalid item name.")
        return

    item = normalize_item_name(item_raw)

    try:
        qty = int(input("Enter quantity: ").strip())
    except Exception:
        print("Invalid quantity.")
        return

    if qty <= 0:
        print("Quantity must be positive.")
        return

    order[item] = order.get(item, 0) + qty
    print(f"Added '{item}: {qty}'.")


def review_and_correct_order(order: Dict[str, int], customer_id: int) -> Optional[Dict[str, int]]:
    print(f"\n========== REVIEW CUSTOMER {customer_id} ORDER ==========")

    while True:
        print_single_order_for_review(order)

        print("\nOptions:")
        print("1 = Accept this order")
        print("2 = Edit an item")
        print("3 = Delete an item")
        print("4 = Add a new item")
        print("5 = Clear all items")
        print("6 = Re-enter this customer's order from input again")

        choice = input("Choose: ").strip()

        if choice == "1":
            print(f"Customer {customer_id} order finalized.")
            return order
        elif choice == "2":
            edit_existing_item(order)
        elif choice == "3":
            delete_existing_item(order)
        elif choice == "4":
            add_new_item(order)
        elif choice == "5":
            order.clear()
            print("Order cleared.")
        elif choice == "6":
            print("You chose to re-enter this customer's order.")
            return None
        else:
            print("Invalid choice.")


# =========================
# ORDER COLLECTION
# =========================
def take_one_customer_order(customer_id: int) -> Dict[str, int]:
    while True:
        print(f"\nCustomer {customer_id} order input mode:")
        print("1 = manual")
        print("2 = scanner/image")
        print("3 = voice")
        mode = input("Choose mode: ").strip()

        if mode == "1":
            order = manual_order_input()

            if not order:
                print("No items entered. Try again.")
                continue

            print(f"\nCustomer {customer_id} manual order: {order}")
            return order

        elif mode == "2":
            order = auto_scan_paper_and_get_order()

            print(f"\nCustomer {customer_id} parsed image order: {order}")

            reviewed = review_and_correct_order(order, customer_id)
            if reviewed is None:
                print(f"Re-entering Customer {customer_id} order...")
                continue

            return reviewed

        elif mode == "3":
            order = record_voice_and_get_order()

            if not order:
                print("No supported items found from voice order. Please try again.")
                continue

            print(f"\nCustomer {customer_id} final voice order: {order}")
            return order

        else:
            print("Invalid mode. Try again.")


# =========================
# SESSION MANAGER
# =========================
class SessionManager:
    def __init__(self, max_baskets: int = 3):
        self.max_baskets = max_baskets
        self.basket_slots = {basket_id: None for basket_id in range(1, max_baskets + 1)}
        self.active_customers: Dict[int, Dict] = {}
        self.waiting_queue = deque()
        self.completed_customers = []
        self.next_customer_id = 1

    def _make_session(self, order: Dict[str, int]) -> Dict:
        customer_id = self.next_customer_id
        self.next_customer_id += 1

        delivered = {item: 0 for item in order.keys()}

        return {
            "customer_id": customer_id,
            "basket_id": None,
            "items_requested": dict(order),
            "items_delivered": delivered,
            "status": "waiting_for_basket",
        }

    def get_free_basket(self) -> Optional[int]:
        for basket_id, assigned_customer in self.basket_slots.items():
            if assigned_customer is None:
                return basket_id
        return None

    def add_customer_order(self, order: Dict[str, int]) -> Dict:
        session = self._make_session(order)
        free_basket = self.get_free_basket()

        if free_basket is not None:
            session["basket_id"] = free_basket
            session["status"] = "active"
            self.basket_slots[free_basket] = session["customer_id"]
            self.active_customers[session["customer_id"]] = session
            print(f"Customer {session['customer_id']} assigned to Basket {free_basket}.")
        else:
            session["status"] = "waiting"
            self.waiting_queue.append(session)
            print(f"Customer {session['customer_id']} added to waiting queue.")

        return session

    def promote_waiting_customer(self):
        if not self.waiting_queue:
            return None

        free_basket = self.get_free_basket()
        if free_basket is None:
            return None

        session = self.waiting_queue.popleft()
        session["basket_id"] = free_basket
        session["status"] = "active"

        self.basket_slots[free_basket] = session["customer_id"]
        self.active_customers[session["customer_id"]] = session

        print(f"Waiting Customer {session['customer_id']} moved to Basket {free_basket}.")
        return session

    def get_pending_items(self, customer_id: int) -> Dict[str, int]:
        session = self.active_customers.get(customer_id)
        if not session:
            return {}

        pending = {}
        for item, req_qty in session["items_requested"].items():
            delivered_qty = session["items_delivered"].get(item, 0)
            remaining = req_qty - delivered_qty
            if remaining > 0:
                pending[item] = remaining
        return pending

    def is_order_complete(self, customer_id: int) -> bool:
        session = self.active_customers.get(customer_id)
        if not session:
            return False

        for item, req_qty in session["items_requested"].items():
            if session["items_delivered"].get(item, 0) < req_qty:
                return False
        return True

    def complete_customer(self, customer_id: int):
        session = self.active_customers.get(customer_id)
        if not session:
            return

        basket_id = session["basket_id"]
        session["status"] = "completed"
        self.completed_customers.append(session)

        if basket_id is not None:
            self.basket_slots[basket_id] = None

        del self.active_customers[customer_id]

        print(f"Customer {customer_id} completed. Basket {basket_id} is now free.")
        self.promote_waiting_customer()

    def deliver_item(self, customer_id: int, item_name: str, quantity: int = 1):
        session = self.active_customers.get(customer_id)
        if not session:
            print("Customer not active.")
            return False

        item_name = normalize_item_name(item_name)
        if item_name not in session["items_requested"]:
            print(f"Customer {customer_id} did not request '{item_name}'.")
            return False

        req_qty = session["items_requested"][item_name]
        current_delivered = session["items_delivered"].get(item_name, 0)

        if current_delivered >= req_qty:
            print(f"Customer {customer_id} already received all '{item_name}'.")
            return False

        new_delivered = min(req_qty, current_delivered + quantity)
        session["items_delivered"][item_name] = new_delivered

        print(
            f"Delivered {item_name} to Customer {customer_id}. "
            f"{new_delivered}/{req_qty} delivered."
        )

        if self.is_order_complete(customer_id):
            self.complete_customer(customer_id)

        return True

    def get_next_customer_for_item(self, item_name: str) -> Optional[int]:
        item_name = normalize_item_name(item_name)

        for customer_id in sorted(self.active_customers.keys()):
            pending = self.get_pending_items(customer_id)
            if pending.get(item_name, 0) > 0:
                return customer_id

        return None

    def build_product_assignment_preview(self) -> Dict[str, list]:
        product_map = defaultdict(list)

        for customer_id in sorted(self.active_customers.keys()):
            pending = self.get_pending_items(customer_id)
            for item, qty in pending.items():
                for _ in range(qty):
                    product_map[item].append(customer_id)

        return dict(product_map)

    def show_status(self):
        print("\n================ BASKET STATUS ================")
        for basket_id in sorted(self.basket_slots.keys()):
            assigned = self.basket_slots[basket_id]
            if assigned is None:
                print(f"Basket {basket_id}: FREE")
            else:
                print(f"Basket {basket_id}: Customer {assigned}")
        print("================================================")

        print("\n================ ACTIVE CUSTOMERS ================")
        if not self.active_customers:
            print("(none)")
        else:
            for customer_id in sorted(self.active_customers.keys()):
                session = self.active_customers[customer_id]
                pending = self.get_pending_items(customer_id)

                print(f"\nCustomer {customer_id} | Basket {session['basket_id']} | Status: {session['status']}")
                print(f"Requested: {session['items_requested']}")
                print(f"Delivered: {session['items_delivered']}")
                print(f"Pending:   {pending}")
        print("==================================================")

        print("\n================ WAITING QUEUE =================")
        if not self.waiting_queue:
            print("(empty)")
        else:
            for session in self.waiting_queue:
                print(f"Customer {session['customer_id']} | Status: {session['status']} | Order: {session['items_requested']}")
        print("================================================")

        print("\n================ COMPLETED CUSTOMERS =================")
        if not self.completed_customers:
            print("(none)")
        else:
            for session in self.completed_customers:
                print(
                    f"Customer {session['customer_id']} | Basket {session['basket_id']} | "
                    f"Requested: {session['items_requested']} | Delivered: {session['items_delivered']}"
                )
        print("======================================================")

        print("\n================ PRODUCT ASSIGNMENT PREVIEW ================")
        preview = self.build_product_assignment_preview()
        if not preview:
            print("(empty)")
        else:
            for item, customer_list in preview.items():
                print(f"{item} -> {customer_list}")
        print("============================================================\n")


# =========================
# INVENTORY DISPLAY
# =========================
def build_combined_item_list_from_sessions(session_manager: SessionManager) -> Dict[str, int]:
    combined = defaultdict(int)

    for session in session_manager.active_customers.values():
        for item, qty in session["items_requested"].items():
            combined[item] += qty

    for session in session_manager.waiting_queue:
        for item, qty in session["items_requested"].items():
            combined[item] += qty

    return dict(combined)


def check_shortages_from_combined(combined_items: Dict[str, int], inventory_stock: Dict[str, int]):
    shortages = {}

    for item, req_qty in combined_items.items():
        available = inventory_stock.get(item, 0)
        if req_qty > available:
            shortages[item] = {
                "required": req_qty,
                "available": available,
                "short": req_qty - available,
            }

    return shortages


def refill_inventory(inventory_stock: Dict[str, int]):
    print("\n========= INVENTORY REFILL =========")
    item = normalize_item_name(input("Enter item name to refill: ").strip())
    if not item:
        print("Invalid item name.")
        return

    try:
        qty = int(input("Enter refill quantity: ").strip())
    except Exception:
        print("Invalid quantity.")
        return

    if qty <= 0:
        print("Quantity must be positive.")
        return

    inventory_stock[item] = inventory_stock.get(item, 0) + qty
    print(f"Refilled {item} by {qty}. New stock = {inventory_stock[item]}")


def print_inventory(inventory_stock: Dict[str, int]):
    print("\n================ INVENTORY =================")
    for item, qty in inventory_stock.items():
        print(f"- {item}: {qty}")
    print("===========================================\n")


def print_combined_item_list(combined_items: Dict[str, int]):
    print("\n============= COMBINED ITEM LIST =============")
    if not combined_items:
        print("(empty)")
    else:
        for item, qty in combined_items.items():
            support_text = "SUPPORTED" if item in SUPPORTED_ITEMS else "UNSUPPORTED"
            print(f"- {item}: {qty} [{support_text}]")
    print("==============================================\n")


def print_shortages(shortages):
    if not shortages:
        print("No inventory shortage detected.\n")
        return

    print("\n************* SHORTAGE ALERT *************")
    for item, info in shortages.items():
        print(
            f"- {item}: required={info['required']}, "
            f"available={info['available']}, short={info['short']}"
        )
    print("******************************************\n")


# =========================
# ESP32 COMMUNICATION
# =========================
class ESP32Link:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(2.0)
        self.busy = False
        self.last_rx = ""

    def read_messages(self):
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line:
                self.last_rx = line
                print(f"[RX] {line}")

                if line == "DONE":
                    self.busy = False
                elif line == "BUSY":
                    self.busy = True

    def send_route(self, basket_id: int, label: str):
        if self.busy:
            return False

        cmd = f"ROUTE:{basket_id}:{label}\n"
        self.ser.write(cmd.encode())
        self.busy = True
        print(f"[TX] {cmd.strip()}")
        return True

    def close(self):
        self.ser.close()


# =========================
# TEMPORAL DETECTION FILTER
# =========================
class DetectionAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = defaultdict(float)
        self.hits = defaultdict(int)
        self.total_frames = 0
        self.missed_frames = 0

    def update(self, label, conf, area_ratio):
        weight = conf * (0.65 + 0.35 * area_ratio)
        self.scores[label] += weight
        self.hits[label] += 1
        self.total_frames += 1
        self.missed_frames = 0

    def no_detection(self):
        self.missed_frames += 1

    def best_two(self):
        if not self.scores:
            return None, 0.0, None, 0.001

        items = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = items[0]
        if len(items) > 1:
            second_label, second_score = items[1]
        else:
            second_label, second_score = None, 0.001
        return top_label, top_score, second_label, second_score

    def decide(self):
        if self.total_frames < MIN_OBSERVED_FRAMES:
            return None

        top_label, top_score, _, second_score = self.best_two()
        if top_label is None:
            return None

        dominance = top_score / max(second_score, 0.001)
        if top_score >= MIN_ACCUMULATED_SCORE and dominance >= DOMINANCE_RATIO:
            return top_label

        return None


# =========================
# ROI CANDIDATE SELECTION
# =========================
def get_best_candidate(results, img):
    frame_area = img.shape[0] * img.shape[1]
    best = None

    if results.boxes is None:
        return None

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if conf < MIN_CONF_IN_ZONE:
            continue

        label = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)

        cx = x1 + w // 2
        cy = y1 + h // 2

        if not (ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2):
            continue

        area = max(1, w * h)
        area_ratio = min(area / frame_area, 0.35) / 0.35
        rank = conf * area

        candidate = {
            "label": label,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "area_ratio": area_ratio,
            "rank": rank,
        }

        if best is None or candidate["rank"] > best["rank"]:
            best = candidate

    return best


# =========================
# CUSTOMER ENTRY
# =========================
def collect_initial_customers(session_manager: SessionManager):
    print("\n========== INITIAL CUSTOMER ENTRY ==========")

    try:
        n = int(input("How many customers to enter now? ").strip())
    except Exception:
        n = 1

    if n < 1:
        n = 1

    for _ in range(n):
        customer_id_guess = session_manager.next_customer_id
        order = take_one_customer_order(customer_id_guess)
        session_manager.add_customer_order(order)


def add_one_more_customer(session_manager: SessionManager):
    customer_id_guess = session_manager.next_customer_id
    order = take_one_customer_order(customer_id_guess)
    session_manager.add_customer_order(order)


# =========================
# AUTONOMOUS SORTING PIPELINE
# =========================
def run_sorting_pipeline(session_manager: SessionManager):
    cap = cv2.VideoCapture(SORTING_CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError("Sorting camera not opened. Try another camera index or close apps using the camera.")

    model = YOLO(MODEL_PATH)

    esp = None
    if USE_SERIAL:
        try:
            esp = ESP32Link(SERIAL_PORT, BAUD_RATE)
            print("ESP32 connected.")
        except Exception as e:
            print(f"ESP32 connection failed: {e}")
            esp = None

    accumulator = DetectionAccumulator()
    waiting_item_to_clear = False
    clear_counter = 0
    last_action = "IDLE"

    while True:
        success, img = cap.read()
        if not success or img is None:
            continue

        if esp is not None:
            esp.read_messages()

        results = model(img, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)

                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2)
                cvzone.putTextRect(
                    img,
                    f"{label} {conf:.2f}",
                    (max(0, x1), max(35, y1)),
                    scale=0.9,
                    thickness=2,
                    offset=6
                )

        cv2.rectangle(img, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)
        cv2.putText(img, "Decision Zone", (ROI_X1, ROI_Y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        candidate = get_best_candidate(results, img)
        controller_busy = esp.busy if esp is not None else False

        if not controller_busy and not waiting_item_to_clear:
            if candidate is not None:
                label = candidate["label"]
                conf = candidate["conf"]
                area_ratio = candidate["area_ratio"]
                x1, y1, x2, y2 = candidate["bbox"]

                accumulator.update(label, conf, area_ratio)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"Candidate: {label}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                decided_label = accumulator.decide()
                if decided_label is not None:
                    customer_id = session_manager.get_next_customer_for_item(decided_label)

                    if customer_id is None:
                        last_action = f"{decided_label} detected but no active customer needs it"
                        print(last_action)
                        waiting_item_to_clear = True
                        clear_counter = 0
                    else:
                        session = session_manager.active_customers.get(customer_id)
                        basket_id = session["basket_id"]

                        if esp is not None:
                            sent = esp.send_route(basket_id, decided_label)
                        else:
                            sent = True
                            print(f"[SIMULATION] ROUTE:BASKET_{basket_id}:{decided_label}")

                        if sent:
                            session_manager.deliver_item(customer_id, decided_label, 1)
                            last_action = f"{decided_label} -> Customer {customer_id} -> Basket {basket_id}"
                            waiting_item_to_clear = True
                            clear_counter = 0
                        else:
                            last_action = "Controller busy"

            else:
                accumulator.no_detection()
                if accumulator.missed_frames >= CLEAR_FRAMES_TO_RESET:
                    accumulator.reset()

        else:
            if candidate is None:
                clear_counter += 1
            else:
                clear_counter = 0

            if clear_counter >= CLEAR_FRAMES_TO_RESET and not controller_busy:
                waiting_item_to_clear = False
                accumulator.reset()
                clear_counter = 0

        top_label, top_score, second_label, second_score = accumulator.best_two()
        cv2.putText(img, f"Top: {top_label} {top_score:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, f"Second: {second_label} {second_score:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, f"Busy: {controller_busy}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, f"Last: {last_action}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y0 = 160
        for customer_id in sorted(session_manager.active_customers.keys()):
            session = session_manager.active_customers[customer_id]
            pending = session_manager.get_pending_items(customer_id)
            text = f"C{customer_id} B{session['basket_id']} Pending: {pending}"
            cv2.putText(img, text[:90], (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y0 += 25

        cv2.imshow("SmartCart Autonomous Sorting", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print("\n--- CURRENT SESSION STATUS ---")
            session_manager.show_status()

    cap.release()
    cv2.destroyAllWindows()
    if esp is not None:
        esp.close()


# =========================
# MENU
# =========================
def main_menu(session_manager: SessionManager):
    while True:
        combined_items = build_combined_item_list_from_sessions(session_manager)
        shortages = check_shortages_from_combined(combined_items, INVENTORY_STOCK)

        session_manager.show_status()
        print_combined_item_list(combined_items)
        print_inventory(INVENTORY_STOCK)
        print_shortages(shortages)

        print("Choose next action:")
        print("1 = Add one more customer")
        print("2 = Refill inventory")
        print("3 = Start autonomous sorting")
        print("4 = Show status again")
        print("5 = Exit")

        choice = input("Choose: ").strip()

        if choice == "1":
            add_one_more_customer(session_manager)
        elif choice == "2":
            refill_inventory(INVENTORY_STOCK)
        elif choice == "3":
            run_sorting_pipeline(session_manager)
        elif choice == "4":
            continue
        elif choice == "5":
            print("Finished.")
            break
        else:
            print("Invalid choice.")


# =========================
# MAIN
# =========================
def main():
    if GEMINI_API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError("Please paste your real Gemini API key in GEMINI_API_KEY.")

    session_manager = SessionManager(max_baskets=MAX_ACTIVE_CUSTOMERS)
    collect_initial_customers(session_manager)
    main_menu(session_manager)


if __name__ == "__main__":
    main()