{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fe3d3ccf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Overwriting models.py\n"
     ]
    }
   ],
   "source": [
    "%%writefile models.py\n",
    "\n",
    "# Define the linear regression model\n",
    "class LinearRegressionModel(nn.Module):\n",
    "    def __init__(self, input_size):\n",
    "        super(LinearRegressionModel, self).__init__()\n",
    "        self.linear = nn.Linear(input_size, 1)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        return self.linear(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07bd521c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
