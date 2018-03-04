class Object:

    def __init__(self, category, top_left, bottom_right):

        self.category = category

        self.top_left = top_left

        self.bottom_right = bottom_right

    def __str__(self):

        return """
        Class : {}
        Bounding Box : {}, {}
        """.format(self.category, self.top_left, self.bottom_right)
