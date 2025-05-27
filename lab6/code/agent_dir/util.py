from numpy.typing import NDArray
import cv2


def img2video(images: list[NDArray], output_path: str, fps: int = 30):
	assert len(images) > 0

	height, width, _ = images[0].shape
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type:ignore
	video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

	for img in images:
		video_writer.write(img)

	video_writer.release()


def make_frame(frame: NDArray, episode: int, step: int) -> NDArray:
	cv2.putText(frame, f'Episode: {episode}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
	cv2.putText(frame, f'Step: {step}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
	return frame


def moving_average(data: list, size: int) -> list[float]:
	avg = [0.0] * size
	window_sum = sum(data[:size])
	avg.append(window_sum / size)

	for i in range(size, len(data)):
		window_sum += data[i] - data[i - size]
		avg.append(window_sum / size)

	return avg
