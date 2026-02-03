import cv2
import numpy as np

def get_average_color(image, mask):
    """Calculates the average color of the masked area."""
    mean_val = cv2.mean(image, mask=mask)
    return (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))

def main():
    cap = cv2.VideoCapture(0)
    
    background = None
    
    print("--- Controls ---")
    print("Press 'b' to capture the blank paper (Background).")
    print("Press 'q' to quit.")
    print("----------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # Smooth the frame to reduce noise
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

        if background is not None:
            # --- STEP 1: Find Changes (Motion) ---
            diff_frame = cv2.absdiff(background, gray_frame)
            _, motion_mask = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)

            # --- STEP 2: Find Skin (HSV Color Space) ---
            hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
            
            # These ranges generally work for many skin tones in decent lighting
            # Lower bound (0, 30, 60) and Upper bound (20, 150, 255) covers typical skin hues
            lower_skin = np.array([0, 30, 60], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            
            skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
            
            # Dilate skin mask to be safe (make the hand "blocker" slightly larger)
            skin_mask = cv2.dilate(skin_mask, None, iterations=2)

            # --- STEP 3: Subtract Skin from Motion ---
            # Logic: (Motion Mask) AND (NOT Skin Mask)
            # We invert the skin mask (255 - skin_mask) so skin becomes black (0)
            skin_inverted = cv2.bitwise_not(skin_mask)
            
            # Combine: Keep motion only where there is NO skin
            paint_mask = cv2.bitwise_and(motion_mask, motion_mask, mask=skin_inverted)

            # --- STEP 4: Analyze the Result ---
            contours, _ = cv2.findContours(paint_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Ignore small specks of noise
                if cv2.contourArea(contour) < 500:
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                
                # Create a specific mask for this detected object to sample color
                object_mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(object_mask, [contour], -1, 255, -1)
                
                # Get the color!
                avg_bgr = get_average_color(frame, object_mask)
                
                # Draw Visuals
                # Green Box = Paint / Brush Tip
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Show color swatch rectangle
                cv2.rectangle(display_frame, (x, y - 30), (x + 50, y), avg_bgr, -1)
                cv2.putText(display_frame, f"Paint", (x + 55, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show debugging windows (Optional but helpful)
            # cv2.imshow("Skin Mask", skin_mask)
            # cv2.imshow("Motion Mask", motion_mask)
            cv2.imshow("Final Paint Mask", paint_mask)

        cv2.imshow("Real-Time Detector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            background = gray_frame
            print("Background captured!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()