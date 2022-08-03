import os
import cv2
import json

class App(object):

    def __init__(self, folder, images, saveto):
        self.data = dict()

        self.saveto = saveto

        self.idx = 0
        self.rect_id = 0
        self.rect_width = 100
        self.rect_height = 100
        self.rects = [None, None]

        self.images = images
        self.images = [os.path.join(folder, img) for img in self.images]
        self.image = cv2.imread(self.images[self.idx])
        self.image = cv2.resize(self.image, (768, 512))

        self.x = 0
        self.y = 0

        cv2.namedWindow('win')
        cv2.setMouseCallback('win', self.onmouse)

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pass
        else:
            self.x = x
            self.y = y

    def save(self):
        with open(self.saveto, 'w') as outfile:
            json.dump(self.data, outfile)


    def run(self):
        while True:
            disp = self.image.copy()

            colors = ((0, 0, 255), (0, 255, 0))
            for rect, color in zip(self.rects, colors):
                if rect is None:
                    continue
                x, y, w, h = rect
                disp = cv2.rectangle(disp, (x, y), (x+w, y+h), color=color, thickness=2)

            disp = cv2.rectangle(disp,
                                 (self.x,
                                  self.y),
                                 (self.x + self.rect_width,
                                  self.y + self.rect_height),
                                 color=(178, 178, 178),
                                 thickness=3)
            cv2.imshow('win', disp)

            ch = cv2.waitKey(5)
            if ch == 27:
                break
            elif ch == ord('1'):
                self.rects[0] = (self.x, self.y, self.rect_width, self.rect_height)
            elif ch == ord('2'):
                self.rects[1] = (self.x, self.y, self.rect_width, self.rect_height)
            elif ch == ord('s'):
                self.save()
                print('Wrote to a file')
            elif ch == ord('n'):
                self.data[os.path.basename(self.images[self.idx])] = self.rects
                self.idx += 1

                if self.idx == len(self.images):
                    self.save()
                    break
                self.rects = [None, None]
                self.image = cv2.imread(self.images[self.idx])
                self.image = cv2.resize(self.image, (768, 512))

        cv2.destroyAllWindows()


if __name__ == '__main__':
    with open('images.json', 'r') as infile:
        data = json.load(infile)

    for key, dataset in data.items():
        if key != 'snow':
            continue
        print('Dataset: {}'.format(key))
        path = "/Users/franz/devel/deweather/data/test/{}/input".format(key)
        app = App(path, dataset['images'], saveto="{}.json".format(key))
        app.run()
