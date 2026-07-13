"""Minimized screen-CV reference for the VizDoom D3 article experiment.

This is an independent GenesisBench implementation of the published behavior.
No upstream source file is vendored.
"""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


def _move(
    *,
    attack: float = 0.0,
    speed: float = 0.0,
    forward: float = 0.0,
    backward: float = 0.0,
    right: float = 0.0,
    left: float = 0.0,
    turn180: float = 0.0,
    turn: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        [attack, speed, forward, backward, right, left, turn180, turn],
        dtype=np.float64,
    )


def _enemy_box(
    image: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    height, width = image.shape[:2]
    scale_x = width / 320.0
    scale_y = height / 240.0
    center = width / 2.0
    channels = [
        image[..., channel].astype(np.int16) for channel in range(3)
    ]
    primary, secondary, tertiary = channels
    enemy_pixels = (
        (primary > 50)
        & ((primary - np.maximum(secondary, tertiary)) > 25)
        & (secondary < 55)
        & (tertiary < 55)
    )
    mask = enemy_pixels.astype(np.uint8) * 255
    mask[: int(40 * scale_y), :] = 0
    mask[int(205 * scale_y) :, :] = 0
    closing_size = max(2, int(round(4 * scale_x)))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((closing_size, closing_size), dtype=np.uint8),
    )
    mask = cv2.dilate(
        mask,
        np.ones(
            (
                max(2, int(4 * scale_y)),
                max(2, int(3 * scale_x)),
            ),
            dtype=np.uint8,
        ),
        iterations=1,
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best: tuple[float, float, float, float, float] | None = None
    best_score = -1.0
    for component in range(1, count):
        _, _, box_width, box_height, area = stats[component]
        center_x, center_y = centroids[component]
        if area < 12 * scale_x * scale_y:
            continue
        if box_height < 7 * scale_y:
            continue
        if box_width > 95 * scale_x or box_height > 115 * scale_y:
            continue
        score = float(area) * (
            float(box_height) / (16 * scale_y)
        ) ** 1.25
        score *= 1.0 + (
            float(center_y) - 90 * scale_y
        ) / (180 * scale_y)
        score /= 1.0 + abs(float(center_x) - center) / (80 * scale_x)
        if score > best_score:
            best_score = score
            best = (
                float(center_x),
                float(center_y),
                float(box_width),
                float(box_height),
                float(area),
            )
    return best


def _supply_box(
    image: np.ndarray,
    *,
    ammo: float,
    health: float,
) -> tuple[float, float, float, float, float] | None:
    height, width = image.shape[:2]
    scale_x = width / 320.0
    scale_y = height / 240.0
    center = width / 2.0
    first = image[..., 0].astype(np.int16)
    second = image[..., 1].astype(np.int16)
    third = image[..., 2].astype(np.int16)
    enemy = (
        (first > 50)
        & ((first - np.maximum(second, third)) > 25)
        & (second < 55)
        & (third < 55)
    )

    medikit_threshold = 95 if ammo <= 5 or health <= -1 else 115
    medikit = (
        (first > medikit_threshold)
        & (second > 75)
        & (third > 70)
    )
    clip = (first > 90) & (second > 60) & ((first - third) > 25)
    bright_mask = ((medikit | clip) & (~enemy)).astype(np.uint8) * 255
    bright_mask[: int(50 * scale_y), :] = 0
    bright_mask[int(218 * scale_y) :, :] = 0
    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (
                max(2, int(3 * scale_y)),
                max(2, int(3 * scale_x)),
            ),
            dtype=np.uint8,
        ),
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        bright_mask,
        8,
    )
    best: tuple[float, float, float, float, float] | None = None
    best_score = -1.0
    for component in range(1, count):
        _, _, box_width, box_height, area = stats[component]
        center_x, center_y = centroids[component]
        if area < 10 * scale_x * scale_y:
            continue
        if area > 3000 * scale_x * scale_y:
            continue
        if box_width > 60 * scale_x or box_height > 52 * scale_y:
            continue
        if box_width < 2 * scale_x or box_height < 2 * scale_y:
            continue
        score = float(area) * (
            1.0 + (float(center_y) - 85 * scale_y) / (130 * scale_y)
        )
        score /= 1.0 + abs(float(center_x) - center) / (85 * scale_x)
        if score > best_score:
            best_score = score
            best = (
                float(center_x),
                float(center_y),
                float(box_width),
                float(box_height),
                float(area),
            )
    if best is not None:
        return best

    color_difference = np.maximum.reduce(
        [
            np.abs(first - second),
            np.abs(first - third),
            np.abs(second - third),
        ]
    )
    fallback_mask = (
        (color_difference > 24) & (~enemy)
    ).astype(np.uint8) * 255
    fallback_mask[: int(55 * scale_y), :] = 0
    fallback_mask[int(205 * scale_y) :, :] = 0
    fallback_mask = cv2.morphologyEx(
        fallback_mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (
                max(2, int(3 * scale_y)),
                max(2, int(3 * scale_x)),
            ),
            dtype=np.uint8,
        ),
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        fallback_mask,
        8,
    )
    best = None
    best_score = -1.0
    for component in range(1, count):
        _, _, box_width, box_height, area = stats[component]
        center_x, center_y = centroids[component]
        if area < 8 * scale_x * scale_y:
            continue
        if area > 700 * scale_x * scale_y:
            continue
        if box_width > 52 * scale_x or box_height > 42 * scale_y:
            continue
        score = float(area) * (
            1.0 + (float(center_y) - 80 * scale_y) / (140 * scale_y)
        )
        score /= 1.0 + abs(float(center_x) - center) / (90 * scale_x)
        if score > best_score:
            best_score = score
            best = (
                float(center_x),
                float(center_y),
                float(box_width),
                float(box_height),
                float(area),
            )
    return best


def _view_key(image: np.ndarray) -> bytes:
    gray = image.astype(np.uint8).mean(axis=2).astype(np.uint8)
    small = cv2.resize(gray, (8, 6), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).astype(np.uint8).tobytes()


class Policy:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.damage = 0.0
        self.hit = 0.0
        self.health = 100.0
        self.lock = 0
        self.bad_lock_steps = 0
        self.last_horizontal_error = 0.0
        self.search_direction = 1.0
        self.panic_steps = 0
        self.previous_frame: np.ndarray | None = None
        self.stuck_steps = 0
        self.last_progress_step = 0
        self.adapted = False
        self.adapted_arc: float | None = None
        self.adapted_bored_after: int | None = None
        self.adapted_bored_turn: int | None = None
        self.escape_index = 0
        self.recent_views: list[bytes] = []

    def _novelty_escape(self) -> np.ndarray:
        self.escape_index += 1
        fraction = (self.escape_index * 0.6180339887498949) % 1.0
        magnitude = 3.0 + fraction * 7.0
        self.search_direction *= -1.0
        return _move(
            speed=1.0,
            backward=1.0,
            turn=self.search_direction * magnitude,
        )

    def act(
        self,
        frame: np.ndarray,
        variables: Mapping[str, float],
    ) -> np.ndarray:
        image = np.asarray(frame)
        height, width = image.shape[:2]
        scale_x = width / 320.0
        scale_y = height / 240.0
        screen_center = width / 2.0
        ammo = float(variables["AMMO2"])
        health = float(variables["HEALTH"])
        damage = float(variables["DAMAGECOUNT"])
        hit = float(variables["HITCOUNT"])
        step = self.step
        self.step += 1

        if damage > self.damage or hit > self.hit:
            self.lock = 10
            self.bad_lock_steps = 0
            self.last_progress_step = step
        elif self.lock > 0:
            self.bad_lock_steps += 1
        if health < self.health - 0.1:
            self.panic_steps = 12
            self.search_direction *= -1.0
        self.damage = damage
        self.hit = hit
        self.health = health

        enemy = _enemy_box(image)
        if enemy is not None and ammo > 0:
            center_x, _, _, box_height, area = enemy
            horizontal_error = (center_x - screen_center) / scale_x
            self.last_horizontal_error = horizontal_error
            self.lock = max(self.lock, 4)
            turn = float(
                np.clip(horizontal_error * 0.18, -8.0, 8.0)
            )
            if abs(horizontal_error) > 4.0:
                danger = (
                    area > 220 * scale_x * scale_y
                    or box_height > 29 * scale_y
                )
                if danger:
                    if horizontal_error < 0:
                        return _move(
                            speed=1.0,
                            backward=1.0,
                            right=1.0,
                            turn=turn,
                        )
                    return _move(
                        speed=1.0,
                        backward=1.0,
                        left=1.0,
                        turn=turn,
                    )
                return _move(speed=1.0, forward=1.0, turn=turn)

            strafe_right = (step // 8) % 2 == 1
            if (
                area > 380 * scale_x * scale_y
                or box_height > 42 * scale_y
                or self.panic_steps > 0
            ):
                self.panic_steps = max(0, self.panic_steps - 1)
                return _move(
                    attack=1.0,
                    speed=1.0,
                    backward=1.0,
                    right=1.0 if strafe_right else 0.0,
                    left=0.0 if strafe_right else 1.0,
                    turn=turn,
                )
            return _move(
                attack=1.0,
                speed=1.0,
                right=1.0 if strafe_right else 0.0,
                left=0.0 if strafe_right else 1.0,
                turn=turn,
            )

        if self.lock > 0 and ammo > 0:
            self.lock -= 1
            turn = float(
                np.clip(
                    self.last_horizontal_error * 0.18,
                    -8.0,
                    8.0,
                )
            )
            if (
                abs(self.last_horizontal_error) <= 4.0
                and self.bad_lock_steps < 4
            ):
                return _move(attack=1.0, turn=turn)
            if self.bad_lock_steps >= 4:
                self.search_direction *= -1.0
                self.bad_lock_steps = 0
            if abs(self.last_horizontal_error) > 4.0:
                return _move(speed=1.0, turn=turn)
            return _move(
                speed=1.0,
                turn=self.search_direction * 6.0,
            )

        if self.panic_steps > 0:
            self.panic_steps -= 1
            return _move(
                speed=1.0,
                backward=1.0,
                turn=self.search_direction * 6.0,
            )

        if ammo <= 10:
            supply = _supply_box(image, ammo=ammo, health=health)
            if supply is not None:
                center_x = supply[0]
                horizontal_error = (center_x - screen_center) / scale_x
                turn = float(
                    np.clip(horizontal_error * 0.18, -8.0, 8.0)
                )
                return _move(speed=1.0, forward=1.0, turn=turn)

        key = _view_key(image)
        repeated = key in self.recent_views
        self.recent_views.append(key)
        if len(self.recent_views) > 80:
            self.recent_views.pop(0)
        if repeated and step - self.last_progress_step > 160:
            return self._novelty_escape()

        if self.previous_frame is not None:
            difference = float(
                np.mean(
                    np.abs(
                        image.astype(np.int16)
                        - self.previous_frame.astype(np.int16)
                    )
                )
            )
            if difference < 0.9:
                self.stuck_steps += 1
            else:
                self.stuck_steps = max(0, self.stuck_steps - 1)
        self.previous_frame = image.copy()
        if self.stuck_steps > 10:
            self.stuck_steps = 0
            self.search_direction *= -1.0
            return _move(
                speed=1.0,
                backward=1.0,
                turn=self.search_direction * 6.0,
            )

        if not self.adapted and step >= 250:
            low_progress = damage <= 15 and health >= 90
            danger_progress = health <= 75 and damage >= 60
            if low_progress or danger_progress:
                self.adapted_arc = 1.4
                self.adapted_bored_after = 180
                self.adapted_bored_turn = 20
            self.adapted = True

        arc_turn = 1.8 if self.adapted_arc is None else self.adapted_arc
        bored_after = (
            190
            if self.adapted_bored_after is None
            else self.adapted_bored_after
        )
        bored_turn = (
            20
            if self.adapted_bored_turn is None
            else self.adapted_bored_turn
        )
        bored_arc = arc_turn
        if damage >= 80 and ammo >= 20 and health >= 85:
            bored_arc = 0.8

        if step - self.last_progress_step > bored_after:
            bored_phase = (
                step - self.last_progress_step - bored_after
            ) % (bored_turn + 90)
            if bored_phase < bored_turn:
                if bored_phase == 0:
                    self.search_direction *= -1.0
                    return _move(speed=1.0, turn180=1.0)
                return _move(
                    speed=1.0,
                    turn=self.search_direction * 6.0,
                )
            return _move(
                speed=1.0,
                forward=1.0,
                turn=self.search_direction * bored_arc,
            )

        return _move(
            speed=1.0,
            forward=1.0,
            turn=self.search_direction * arc_turn,
        )
