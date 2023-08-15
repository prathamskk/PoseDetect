/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java.posedetector;

import static java.lang.Math.atan2;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.text.TextUtils;

import androidx.annotation.Nullable;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;
import com.google.mlkit.vision.demo.InferenceInfoGraphic;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;
import java.util.List;
import java.util.Locale;

/** Draw the detected pose in preview. */
public class PoseGraphic extends Graphic {

  private static final float DOT_RADIUS = 8.0f;
  private static final float IN_FRAME_LIKELIHOOD_TEXT_SIZE = 30.0f;

  private final Pose pose;
  private final boolean showInFrameLikelihood;
  private final Paint leftPaint;
  private final Paint rightPaint;
  private final Paint whitePaint;
  private final Paint tipPaint;

  PoseGraphic(GraphicOverlay overlay, Pose pose, boolean showInFrameLikelihood) {
    super(overlay);

    this.pose = pose;
    this.showInFrameLikelihood = showInFrameLikelihood;

    whitePaint = new Paint();
    whitePaint.setColor(Color.WHITE);
    whitePaint.setTextSize(IN_FRAME_LIKELIHOOD_TEXT_SIZE);
    leftPaint = new Paint();
    leftPaint.setColor(Color.GREEN);
    rightPaint = new Paint();
    rightPaint.setColor(Color.YELLOW);
    tipPaint = new Paint();
    tipPaint.setColor(Color.WHITE);
    tipPaint.setTextSize(40f);
  }

  @Override
  public void draw(Canvas canvas) {
    List<PoseLandmark> landmarks = pose.getAllPoseLandmarks();
    if (landmarks.isEmpty()) {
      return;
    }
    // Draw all the points
    for (PoseLandmark landmark : landmarks) {
      drawPoint(canvas, landmark.getPosition(), whitePaint);
      if (showInFrameLikelihood) {
        canvas.drawText(
            String.format(Locale.US, "%.2f", landmark.getInFrameLikelihood()),
            translateX(landmark.getPosition().x),
            translateY(landmark.getPosition().y),
            whitePaint);
      }
    }
    PoseLandmark leftShoulder = pose.getPoseLandmark(PoseLandmark.Type.LEFT_SHOULDER);
    PoseLandmark rightShoulder = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_SHOULDER);
    PoseLandmark leftElbow = pose.getPoseLandmark(PoseLandmark.Type.LEFT_ELBOW);
    PoseLandmark rightElbow = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_ELBOW);
    PoseLandmark leftWrist = pose.getPoseLandmark(PoseLandmark.Type.LEFT_WRIST);
    PoseLandmark rightWrist = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_WRIST);
    PoseLandmark leftHip = pose.getPoseLandmark(PoseLandmark.Type.LEFT_HIP);
    PoseLandmark rightHip = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_HIP);
    PoseLandmark leftKnee = pose.getPoseLandmark(PoseLandmark.Type.LEFT_KNEE);
    PoseLandmark rightKnee = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_KNEE);
    PoseLandmark leftAnkle = pose.getPoseLandmark(PoseLandmark.Type.LEFT_ANKLE);
    PoseLandmark rightAnkle = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_ANKLE);

    PoseLandmark leftPinky = pose.getPoseLandmark(PoseLandmark.Type.LEFT_PINKY);
    PoseLandmark rightPinky = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_PINKY);
    PoseLandmark leftIndex = pose.getPoseLandmark(PoseLandmark.Type.LEFT_INDEX);
    PoseLandmark rightIndex = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_INDEX);
    PoseLandmark leftThumb = pose.getPoseLandmark(PoseLandmark.Type.LEFT_THUMB);
    PoseLandmark rightThumb = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_THUMB);
    PoseLandmark leftHeel = pose.getPoseLandmark(PoseLandmark.Type.LEFT_HEEL);
    PoseLandmark rightHeel = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_HEEL);
    PoseLandmark leftFootIndex = pose.getPoseLandmark(PoseLandmark.Type.LEFT_FOOT_INDEX);
    PoseLandmark rightFootIndex = pose.getPoseLandmark(PoseLandmark.Type.RIGHT_FOOT_INDEX);
    // Calculate whether the hand exceeds the shoulder
    float yRightHand = rightWrist.getPosition().y - rightShoulder.getPosition().y;
    float yLeftHand = leftWrist.getPosition().y - leftShoulder.getPosition().y;

// Calculate whether the distance between the shoulder and the foot is the same width
    float shoulderDistance = leftShoulder.getPosition().x - rightShoulder.getPosition().x;
    float footDistance = leftAnkle.getPosition().x - rightAnkle.getPosition().x;
    float ratio = footDistance / shoulderDistance;

// Angle of point 24-26-28
    double angle24_26_28 = getAngle(rightHip, rightKnee, rightAnkle);

    if (((180 - Math.abs(angle24_26_28)) > 5) && !isCount) {
      reInitParams();
      lineOneText = "Please stand up straight";
    } else if (yLeftHand > 0 || yRightHand > 0) {
      reInitParams();
      lineOneText = "Please hold your hands behind your head";
    } else if (ratio < 0.5 && !isCount) {
      reInitParams();
      lineOneText = "Please spread your feet shoulder-width apart";
    } else {
      float currentHeight = (rightShoulder.getPosition().y + leftShoulder.getPosition().y) / 2;

      if (!isCount) {
        shoulderHeight = currentHeight;
        minSize = (rightAnkle.getPosition().y - rightHip.getPosition().y) / 5;
        isCount = true;
        lastHeight = currentHeight;
        lineOneText = "Gesture ready";
      }
      if (!isDown && (currentHeight - lastHeight) > minSize) {
        isDown = true;
        isUp = false;
        downCount++;
        lastHeight = currentHeight;
        lineTwoText = "start down";
      } else if ((currentHeight - lastHeight) > minSize) {
        lineTwoText = "downing";
        lastHeight = currentHeight;
      }
      if (!isUp && (upCount < downCount) && (lastHeight - currentHeight) > minSize) {
        isUp = true;
        isDown = false;
        upCount++;
        lastHeight = currentHeight;
        lineTwoText = "start up";
      } else if ((lastHeight - currentHeight) > minSize) {
        lineTwoText = "uping";
        lastHeight = currentHeight;
      }
    }

    drawText(canvas, lineOneText, 1);
    drawText(canvas, lineTwoText, 2);
    drawText(canvas, "count: " + upCount, 3);

    drawLine(canvas, leftShoulder.getPosition(), rightShoulder.getPosition(), whitePaint);
    drawLine(canvas, leftHip.getPosition(), rightHip.getPosition(), whitePaint);

    // Left body
    drawLine(canvas, leftShoulder.getPosition(), leftElbow.getPosition(), leftPaint);
    drawLine(canvas, leftElbow.getPosition(), leftWrist.getPosition(), leftPaint);
    drawLine(canvas, leftShoulder.getPosition(), leftHip.getPosition(), leftPaint);
    drawLine(canvas, leftHip.getPosition(), leftKnee.getPosition(), leftPaint);
    drawLine(canvas, leftKnee.getPosition(), leftAnkle.getPosition(), leftPaint);
    drawLine(canvas, leftWrist.getPosition(), leftThumb.getPosition(), leftPaint);
    drawLine(canvas, leftWrist.getPosition(), leftPinky.getPosition(), leftPaint);
    drawLine(canvas, leftWrist.getPosition(), leftIndex.getPosition(), leftPaint);
    drawLine(canvas, leftAnkle.getPosition(), leftHeel.getPosition(), leftPaint);
    drawLine(canvas, leftHeel.getPosition(), leftFootIndex.getPosition(), leftPaint);

    // Right body
    drawLine(canvas, rightShoulder.getPosition(), rightElbow.getPosition(), rightPaint);
    drawLine(canvas, rightElbow.getPosition(), rightWrist.getPosition(), rightPaint);
    drawLine(canvas, rightShoulder.getPosition(), rightHip.getPosition(), rightPaint);
    drawLine(canvas, rightHip.getPosition(), rightKnee.getPosition(), rightPaint);
    drawLine(canvas, rightKnee.getPosition(), rightAnkle.getPosition(), rightPaint);
    drawLine(canvas, rightWrist.getPosition(), rightThumb.getPosition(), rightPaint);
    drawLine(canvas, rightWrist.getPosition(), rightPinky.getPosition(), rightPaint);
    drawLine(canvas, rightWrist.getPosition(), rightIndex.getPosition(), rightPaint);
    drawLine(canvas, rightAnkle.getPosition(), rightHeel.getPosition(), rightPaint);
    drawLine(canvas, rightHeel.getPosition(), rightFootIndex.getPosition(), rightPaint);
  }

  void drawPoint(Canvas canvas, @Nullable PointF point, Paint paint) {
    if (point == null) {
      return;
    }
    canvas.drawCircle(translateX(point.x), translateY(point.y), DOT_RADIUS, paint);
  }

  void drawLine(Canvas canvas, @Nullable PointF start, @Nullable PointF end, Paint paint) {
    if (start == null || end == null) {
      return;
    }
    canvas.drawLine(
        translateX(start.x), translateY(start.y), translateX(end.x), translateY(end.y), paint);
  }

  public void drawText(Canvas canvas, String text, int line) {
    if (TextUtils.isEmpty(text)) {
      return;
    }
    canvas.drawText(text, InferenceInfoGraphic.TEXT_SIZE * 0.5f,
            InferenceInfoGraphic.TEXT_SIZE * 3 + InferenceInfoGraphic.TEXT_SIZE * line, tipPaint);
  }

  private static boolean isUp = false;
  private static boolean isDown = false;
  private static int upCount = 0;
  private static int downCount = 0;
  private static boolean isCount = false;
  private static String lineOneText = "";
  private static String lineTwoText = "";
  private static float shoulderHeight = 0f;
  private static float minSize = 0f;
  private static float lastHeight = 0f;
  public void reInitParams() {
    lineOneText = "";
    lineTwoText = "";
    shoulderHeight = 0f;
    minSize = 0f;
    isCount = false;
    isUp = false;
    isDown = false;
    upCount = 0;
    downCount = 0;
  }
  private double getAngle(PoseLandmark firstPoint, PoseLandmark midPoint, PoseLandmark lastPoint) {
    double result = Math.toDegrees(atan2(1.0 * lastPoint.getPosition().y - midPoint.getPosition().y,
            1.0 * lastPoint.getPosition().x - midPoint.getPosition().x)
            - atan2(firstPoint.getPosition().y - midPoint.getPosition().y,
            firstPoint.getPosition().x - midPoint.getPosition().x));
    result = Math.abs(result);
    if (result > 180) {
      result = 360.0 - result;
    }
    return result;
  }
}
