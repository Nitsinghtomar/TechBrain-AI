import React from 'react';
import { render } from '@testing-library/react';
import App from './App';

test('renders learn react link', () => {
  // Destructure the methods from the result of the render function
  const { getByText } = render(<App />);

  // Use getByText directly from the render result
  const linkElement = getByText(/learn react/i);

  expect(linkElement).toBeInTheDocument();
});
