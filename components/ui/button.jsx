import { Spinner } from './Spinner'; 

export const Button = ({ 
  children, 
  isLoading = false, 
  className = '', 
  variant = 'primary',
  ...props 
}) => {
  const variantClasses = {
    primary: 'primary-btn',
    secondary: 'outline-btn',
    outline: 'outline-btn',
  };

  const buttonClass = variantClasses[variant] || variantClasses.primary;

  return (
    <button
      // 4. Use the dynamic buttonClass in the className string
      className={`w-full flex justify-center ${buttonClass} ${className}`}
      disabled={isLoading}
      {...props}
    >
      {isLoading ? <Spinner /> : children}
    </button>
  );
};