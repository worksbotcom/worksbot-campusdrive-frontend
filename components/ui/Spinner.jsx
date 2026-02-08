// components/ui/Spinner.jsx
export const Spinner = ({ size = 'sm' }) => {
    const sizeClasses = { sm: 'h-5 w-5', md: 'h-8 w-8', lg: 'h-12 w-12' };
    return (
        <div className={`animate-spin rounded-full border-t-2 border-r-2 border-primary ${sizeClasses[size]}`}></div>
    );
};